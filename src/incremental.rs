/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Region updates for a fixed reference.

use crate::metric::ReferenceData;
use crate::pyramid::{Level, PackedLevel, Plane, Region, TilePlane, Workspace};
use crate::{Error, GrayImage, MAX_SCALES, PreparedReference, Score};
use std::ops::Range;
use std::sync::{Arc, Weak};

/// A half-open pixel rectangle `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self { x0, y0, x1, y1 }
    }

    fn clipped(self, width: usize, height: usize) -> Option<Region> {
        let x1 = self.x1.min(width);
        let y1 = self.y1.min(height);
        if self.x0 >= x1 || self.y0 >= y1 {
            return None;
        }
        Some(Region {
            x: self.x0,
            y: self.y0,
            width: x1 - self.x0,
            height: y1 - self.y0,
        })
    }
}

/// Saved pixels and scores for one trial update. Reuse the checkpoint across
/// calls to [IncrementalState::update_checkpointed] or
/// [IncrementalState::update_patch_checkpointed] to retain its buffers.
#[derive(Default)]
pub struct Checkpoint {
    update: Option<SavedUpdate>,
    bytes: Vec<u8>,
    floats: Vec<f32>,
    sums: Vec<f64>,
}

struct SavedUpdate {
    owner: Weak<()>,
    before: u64,
    after: u64,
    region: Region,
}

impl Checkpoint {
    /// Bytes reserved by the checkpoint's pixel and score buffers.
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.capacity()
            + self.floats.capacity() * size_of::<f32>()
            + self.sums.capacity() * size_of::<f64>()
    }
}

/// A fixed reference and the current candidate, scored over all scales.
/// Only the candidate's finest samples are retained. Updates rebuild the
/// small, independent tiles that intersect the changed rectangle.
pub struct IncrementalState {
    reference: Arc<ReferenceData>,
    candidate: Level,
    workspace: Workspace,
    grid: TileGrid,
    sums: TileSums,
    identity: Arc<()>,
    revision: u64,
    next_revision: u64,
}

impl IncrementalState {
    /// Score `test` in full and retain its samples for later updates.
    pub fn new(reference: &PreparedReference, test: GrayImage<'_>) -> Result<Self, Error> {
        let reference = &reference.data;
        if (test.width(), test.height()) != (reference.width, reference.height) {
            return Err(Error::DimensionMismatch);
        }
        let layout = reference.layout;
        let side = layout.alignment();
        let grid = TileGrid {
            side,
            columns: layout.width / side,
            rows: layout.height / side,
        };
        let candidate = test.padded(Region {
            x: 0,
            y: 0,
            width: test.width(),
            height: test.height(),
        })?;
        let mut state = Self {
            reference: Arc::clone(reference),
            candidate,
            workspace: Workspace::with_area(side * side, layout.scales, test.is_byte()),
            grid,
            sums: TileSums::new(grid.columns * grid.rows, layout.scales),
            identity: Arc::new(()),
            revision: 0,
            next_revision: 1,
        };
        state.rescore(grid.full());
        Ok(state)
    }

    pub fn width(&self) -> usize {
        self.reference.width
    }

    pub fn height(&self) -> usize {
        self.reference.height
    }

    pub fn scales(&self) -> usize {
        self.reference.layout.scales
    }

    /// Bytes reserved by the candidate, score and workspace buffers,
    /// excluding the shared prepared reference.
    pub fn allocated_bytes(&self) -> usize {
        let pixels = match &self.candidate {
            Level::Byte(level) => level.samples.capacity(),
            Level::Float(level) => level.samples.capacity() * size_of::<f32>(),
        };
        pixels + self.sums.nodes.capacity() * size_of::<f64>() + self.workspace.allocated_bytes()
    }

    /// The current score. Results agree with a full comparison up to
    /// floating-point summation order.
    pub fn score(&self) -> Score {
        let mut sums = [0.0; MAX_SCALES];
        sums[..self.scales()].copy_from_slice(self.sums.root());
        Score::from_sums(sums, self.reference.layout)
    }

    /// Update the candidate using the pixels in `rect`, clipped to the image.
    /// Pass a full image view with the dimensions and sample type used to
    /// initialize this state. Only pixels in the clipped rectangle are read.
    /// On error the state is unchanged.
    pub fn update(&mut self, test: GrayImage<'_>, rect: Rect) -> Result<(), Error> {
        self.apply(test, rect, None)
    }

    /// Update and save the previous pixels and scores for [Self::rollback].
    /// On error the state and checkpoint are unchanged. An empty update clears
    /// the checkpoint.
    pub fn update_checkpointed(
        &mut self,
        test: GrayImage<'_>,
        rect: Rect,
        checkpoint: &mut Checkpoint,
    ) -> Result<(), Error> {
        self.apply(test, rect, Some(checkpoint))
    }

    /// Replace the candidate pixels at `(x, y)` with `patch`, saving the previous
    /// pixels and scores for [Self::rollback]. The patch may have its own row stride.
    /// A patch outside the candidate returns [Error::DimensionMismatch]. A
    /// different sample type returns [Error::InvalidLayout].
    /// On error the state and checkpoint are unchanged.
    pub fn update_patch_checkpointed(
        &mut self,
        patch: GrayImage<'_>,
        x: usize,
        y: usize,
        checkpoint: &mut Checkpoint,
    ) -> Result<(), Error> {
        let (width, height) = (patch.width(), patch.height());
        if width > self.width().saturating_sub(x) || height > self.height().saturating_sub(y) {
            return Err(Error::DimensionMismatch);
        }
        self.check_format(patch)?;
        let source = Region {
            x: 0,
            y: 0,
            width,
            height,
        };
        let pixels = patch.checked_region(source)?;
        self.apply_pixels(Region { x, y, ..source }, pixels, Some(checkpoint));
        Ok(())
    }

    /// Undo the update that filled `checkpoint`, restoring the previous score
    /// exactly. Checkpoints from this state can be rolled back in reverse
    /// order. A checkpoint for a different revision or state is ignored.
    pub fn rollback(&mut self, checkpoint: &Checkpoint) {
        let Some(saved) = &checkpoint.update else {
            return;
        };
        if saved.after != self.revision
            || !std::ptr::eq(saved.owner.as_ptr(), Arc::as_ptr(&self.identity))
        {
            return;
        }
        let region = saved.region;
        let pixels = match &self.candidate {
            Level::Byte(_) => TilePlane::Byte(Plane::packed(&checkpoint.bytes, region)),
            Level::Float(_) => TilePlane::Float(Plane::packed(&checkpoint.floats, region)),
        };
        self.replace_pixels(region, pixels);
        let tiles = self.grid.covering(region);
        let mut saved_rows = checkpoint
            .sums
            .chunks_exact((tiles.x1 - tiles.x0) * self.scales());
        for y in tiles.y0..tiles.y1 {
            let leaves = self.grid.row(tiles, y);
            self.sums
                .leaves_mut(leaves)
                .copy_from_slice(saved_rows.next().expect("saved tile row"));
        }
        self.refresh_sums(tiles);
        self.revision = saved.before;
    }

    fn apply(
        &mut self,
        test: GrayImage<'_>,
        rect: Rect,
        checkpoint: Option<&mut Checkpoint>,
    ) -> Result<(), Error> {
        if (test.width(), test.height()) != (self.width(), self.height()) {
            return Err(Error::DimensionMismatch);
        }
        self.check_format(test)?;
        let Some(region) = rect.clipped(self.width(), self.height()) else {
            if let Some(checkpoint) = checkpoint {
                checkpoint.update = None;
            }
            return Ok(());
        };
        let pixels = test.checked_region(region)?;
        self.apply_pixels(region, pixels, checkpoint);
        Ok(())
    }

    fn check_format(&self, image: GrayImage<'_>) -> Result<(), Error> {
        if image.is_byte() != matches!(self.candidate, Level::Byte(_)) {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }

    fn apply_pixels(
        &mut self,
        region: Region,
        pixels: TilePlane<'_>,
        checkpoint: Option<&mut Checkpoint>,
    ) {
        let revision = self.next_revision;
        self.next_revision = revision
            .checked_add(1)
            .expect("incremental revision overflow");
        let tiles = self.grid.covering(region);

        if let Some(checkpoint) = checkpoint {
            match self.candidate.region(region) {
                TilePlane::Byte(pixels) => save_pixels(pixels, &mut checkpoint.bytes),
                TilePlane::Float(pixels) => save_pixels(pixels, &mut checkpoint.floats),
            }
            checkpoint.sums.clear();
            for y in tiles.y0..tiles.y1 {
                checkpoint
                    .sums
                    .extend_from_slice(self.sums.leaves(self.grid.row(tiles, y)));
            }
            checkpoint.update = Some(SavedUpdate {
                owner: Arc::downgrade(&self.identity),
                before: self.revision,
                after: revision,
                region,
            });
        }
        self.replace_pixels(region, pixels);
        self.rescore(tiles);
        self.revision = revision;
    }

    fn replace_pixels(&mut self, region: Region, pixels: TilePlane<'_>) {
        match (&mut self.candidate, pixels) {
            (Level::Byte(candidate), TilePlane::Byte(source)) => {
                copy_pixels(candidate, region, source)
            }
            (Level::Float(candidate), TilePlane::Float(source)) => {
                copy_pixels(candidate, region, source)
            }
            _ => unreachable!("incremental sample format was checked"),
        }
    }

    fn rescore(&mut self, tiles: Rect) {
        let image = GrayImage::from_packed(&self.candidate, self.height());
        let scales = self.scales();
        for y in tiles.y0..tiles.y1 {
            let mut scores = self
                .sums
                .leaves_mut(self.grid.row(tiles, y))
                .chunks_exact_mut(scales);
            for x in tiles.x0..tiles.x1 {
                self.reference
                    .score_tile(
                        image,
                        self.grid.tile(x, y),
                        &mut self.workspace,
                        scores.next().expect("tile score"),
                    )
                    .expect("incremental candidate contains only finite samples");
            }
        }
        self.refresh_sums(tiles);
    }

    fn refresh_sums(&mut self, tiles: Rect) {
        if tiles == self.grid.full() {
            self.sums.rebuild();
        } else {
            for y in tiles.y0..tiles.y1 {
                self.sums.refresh(self.grid.row(tiles, y));
            }
        }
    }
}

fn save_pixels<T: Copy>(source: Plane<'_, T>, output: &mut Vec<T>) {
    output.clear();
    output.reserve(source.width * source.height);
    for row in source.samples.chunks(source.stride) {
        output.extend_from_slice(&row[..source.width]);
    }
}

fn copy_pixels<T: Copy>(target: &mut PackedLevel<T>, region: Region, source: Plane<'_, T>) {
    let rows = target.samples.chunks_exact_mut(target.width).skip(region.y);
    for (dst, src) in rows.zip(source.samples.chunks(source.stride)) {
        dst[region.x..region.x + region.width].copy_from_slice(&src[..region.width]);
    }
}

#[derive(Clone, Copy)]
struct TileGrid {
    side: usize,
    columns: usize,
    rows: usize,
}

impl TileGrid {
    fn full(self) -> Rect {
        Rect::new(0, 0, self.columns, self.rows)
    }

    fn covering(self, region: Region) -> Rect {
        Rect::new(
            region.x / self.side,
            region.y / self.side,
            (region.x + region.width).div_ceil(self.side),
            (region.y + region.height).div_ceil(self.side),
        )
    }

    fn row(self, tiles: Rect, y: usize) -> Range<usize> {
        let start = y * self.columns;
        start + tiles.x0..start + tiles.x1
    }

    fn tile(self, x: usize, y: usize) -> Region {
        Region {
            x: x * self.side,
            y: y * self.side,
            width: self.side,
            height: self.side,
        }
    }
}

/// Binary tree of tile sums, with the leaves stored in raster order starting
/// at node `tiles`. Fixed child sums avoid accumulating roundoff over updates.
struct TileSums {
    nodes: Vec<f64>,
    tiles: usize,
    scales: usize,
}

impl TileSums {
    fn new(tiles: usize, scales: usize) -> Self {
        Self {
            nodes: vec![0.0; 2 * tiles * scales],
            tiles,
            scales,
        }
    }

    fn root(&self) -> &[f64] {
        &self.nodes[self.scales..2 * self.scales]
    }

    fn leaf_range(&self, tiles: Range<usize>) -> Range<usize> {
        (self.tiles + tiles.start) * self.scales..(self.tiles + tiles.end) * self.scales
    }

    fn leaves(&self, tiles: Range<usize>) -> &[f64] {
        &self.nodes[self.leaf_range(tiles)]
    }

    fn leaves_mut(&mut self, tiles: Range<usize>) -> &mut [f64] {
        let range = self.leaf_range(tiles);
        &mut self.nodes[range]
    }

    fn rebuild(&mut self) {
        for node in (1..self.tiles).rev() {
            self.sum_children(node);
        }
    }

    fn refresh(&mut self, leaves: Range<usize>) {
        let mut nodes = self.tiles + leaves.start..self.tiles + leaves.end;
        while nodes.start > 1 {
            nodes = nodes.start / 2..nodes.end.div_ceil(2);
            for node in nodes.clone().rev() {
                self.sum_children(node);
            }
        }
    }

    fn sum_children(&mut self, node: usize) {
        let parent = node * self.scales;
        let left = 2 * parent;
        let right = left + self.scales;
        for scale in 0..self.scales {
            self.nodes[parent + scale] = self.nodes[left + scale] + self.nodes[right + scale];
        }
    }
}
