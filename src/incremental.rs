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

use crate::image::check_finite;
use crate::metric::ReferenceData;
use crate::pyramid::{Level, PackedLevel, Plane, Region, TilePlane, WINDOW, Workspace};
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
    region: Option<Region>,
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
    workspaces: [Workspace; 2],
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
        Ok(Self::from_packed(reference, test.to_packed()?))
    }

    /// Score a tightly packed byte candidate and take ownership of its
    /// existing allocation. The vector length must equal the reference's
    /// width times its height.
    pub fn from_u8(reference: &PreparedReference, samples: Vec<u8>) -> Result<Self, Error> {
        let reference = &reference.data;
        let width = reference.width;
        if samples.len() != width * reference.height {
            return Err(Error::InvalidLayout);
        }
        Ok(Self::from_packed(
            reference,
            Level::Byte(PackedLevel { samples, width }),
        ))
    }

    /// Score a tightly packed float candidate and take ownership of its
    /// existing allocation. The vector length must equal the reference's
    /// width times its height. Samples must be finite.
    pub fn from_unit_f32(reference: &PreparedReference, samples: Vec<f32>) -> Result<Self, Error> {
        let reference = &reference.data;
        let width = reference.width;
        if samples.len() != width * reference.height {
            return Err(Error::InvalidLayout);
        }
        check_finite(&samples)?;
        Ok(Self::from_packed(
            reference,
            Level::Float(PackedLevel { samples, width }),
        ))
    }

    fn from_packed(reference: &Arc<ReferenceData>, candidate: Level) -> Self {
        let layout = reference.layout;
        let side = layout.alignment();
        let grid = TileGrid {
            side,
            columns: layout.width / side,
            rows: layout.height / side,
        };
        let mut state = Self {
            reference: Arc::clone(reference),
            candidate,
            workspaces: Default::default(),
            grid,
            sums: TileSums::new(grid.columns * grid.rows, layout.scales),
            identity: Arc::new(()),
            revision: 0,
            next_revision: 1,
        };
        state.rescore(grid.full());
        state
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
        pixels
            + self.sums.nodes.capacity() * size_of::<f64>()
            + self
                .workspaces
                .iter()
                .map(Workspace::allocated_bytes)
                .sum::<usize>()
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
        let Some(region) = saved.region else {
            self.revision = saved.before;
            return;
        };
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
        let previous = self.candidate.region(region);
        let changed = changed_region(previous, pixels).map(|offset| {
            let region = Region {
                x: region.x + offset.x,
                y: region.y + offset.y,
                ..offset
            };
            (region, self.grid.covering(region), offset)
        });

        if let Some(checkpoint) = checkpoint {
            if let Some((region, tiles, _)) = changed {
                match self.candidate.region(region) {
                    TilePlane::Byte(pixels) => save_pixels(pixels, &mut checkpoint.bytes),
                    TilePlane::Float(pixels) => save_pixels(pixels, &mut checkpoint.floats),
                }
                checkpoint.sums.clear();
                checkpoint
                    .sums
                    .reserve((tiles.x1 - tiles.x0) * (tiles.y1 - tiles.y0) * self.scales());
                for y in tiles.y0..tiles.y1 {
                    checkpoint
                        .sums
                        .extend_from_slice(self.sums.leaves(self.grid.row(tiles, y)));
                }
            }
            // An unchanged nonempty update still advances the revision and
            // can be rolled back before an earlier checkpoint.
            checkpoint.update = Some(SavedUpdate {
                owner: Arc::downgrade(&self.identity),
                before: self.revision,
                after: revision,
                region: changed.map(|(region, _, _)| region),
            });
        }
        if let Some((region, tiles, offset)) = changed {
            let pixels = match pixels {
                TilePlane::Byte(pixels) => TilePlane::Byte(pixels.subregion(offset)),
                TilePlane::Float(pixels) => TilePlane::Float(pixels.subregion(offset)),
            };
            self.replace_pixels(region, pixels);
            self.rescore(tiles);
        }
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
        let scales = self.scales();
        for y in tiles.y0..tiles.y1 {
            let mut scores = self
                .sums
                .leaves_mut(self.grid.row(tiles, y))
                .chunks_exact_mut(scales);
            for x in tiles.x0..tiles.x1 {
                self.reference.score_tile(
                    &self.candidate,
                    self.grid.tile(x, y),
                    &mut self.workspaces,
                    scores.next().expect("tile score"),
                );
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

fn changed_region(previous: TilePlane<'_>, proposed: TilePlane<'_>) -> Option<Region> {
    match (previous, proposed) {
        (TilePlane::Byte(a), TilePlane::Byte(b)) => {
            changed_bounds(a, b, |a, b| a == b, |a, b| a == b)
        }
        (TilePlane::Float(a), TilePlane::Float(b)) => {
            let eq = |a: &f32, b: &f32| a.to_bits() == b.to_bits();
            changed_bounds(a, b, |a, b| a.iter().zip(b).all(|(a, b)| eq(a, b)), eq)
        }
        _ => unreachable!("incremental sample format was checked"),
    }
}

fn changed_bounds<T>(
    a: Plane<'_, T>,
    b: Plane<'_, T>,
    rows_equal: impl Fn(&[T], &[T]) -> bool,
    samples_equal: impl Fn(&T, &T) -> bool,
) -> Option<Region> {
    let (width, height) = (a.width, a.height);
    let mut rows = a
        .samples
        .chunks(a.stride)
        .zip(b.samples.chunks(b.stride))
        .map(|(a, b)| (&a[..width], &b[..width]));
    let mut changed_rows = rows.by_ref().enumerate();
    let (top, (a, b)) = changed_rows.find(|(_, (a, b))| !rows_equal(a, b))?;

    if width * height <= WINDOW * WINDOW {
        return Some(Region {
            x: 0,
            y: 0,
            width,
            height,
        });
    }

    let mut differences = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (a, b))| !samples_equal(a, b));
    let mut left = differences.next().expect("changed row").0;
    let mut right = differences.next_back().map_or(left + 1, |(x, _)| x + 1);
    let mut bottom = top + 1;
    if let Some((y, (a, b))) = changed_rows.rfind(|(_, (a, b))| !rows_equal(a, b)) {
        bottom = y + 1;
        for (a, b) in std::iter::once((a, b)).chain(changed_rows.map(|(_, rows)| rows)) {
            if left == 0 && right == width {
                break;
            }
            if !rows_equal(&a[..left], &b[..left]) {
                left = a[..left]
                    .iter()
                    .zip(&b[..left])
                    .position(|(a, b)| !samples_equal(a, b))
                    .expect("changed prefix");
            }
            if !rows_equal(&a[right..], &b[right..]) {
                right += a[right..]
                    .iter()
                    .zip(&b[right..])
                    .rposition(|(a, b)| !samples_equal(a, b))
                    .expect("changed suffix")
                    + 1;
            }
        }
    }
    Some(Region {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
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

/// Four-way tree of tile sums, with the leaves stored in raster order.
/// Fixed child sums avoid accumulating roundoff over updates.
struct TileSums {
    nodes: Vec<f64>,
    first_leaf: usize,
    scales: usize,
}

impl TileSums {
    const CHILDREN: usize = 4;

    fn new(tiles: usize, scales: usize) -> Self {
        // Every complete branch adds three leaves. The final branch may
        // have fewer children, so neither leaves nor branches need padding.
        let first_leaf = (tiles - 1).div_ceil(Self::CHILDREN - 1);
        Self {
            nodes: vec![0.0; (first_leaf + tiles) * scales],
            first_leaf,
            scales,
        }
    }

    fn root(&self) -> &[f64] {
        &self.nodes[..self.scales]
    }

    fn leaf_range(&self, tiles: Range<usize>) -> Range<usize> {
        (self.first_leaf + tiles.start) * self.scales..(self.first_leaf + tiles.end) * self.scales
    }

    fn leaves(&self, tiles: Range<usize>) -> &[f64] {
        &self.nodes[self.leaf_range(tiles)]
    }

    fn leaves_mut(&mut self, tiles: Range<usize>) -> &mut [f64] {
        let range = self.leaf_range(tiles);
        &mut self.nodes[range]
    }

    fn rebuild(&mut self) {
        for node in (0..self.first_leaf).rev() {
            self.sum_children(node);
        }
    }

    fn refresh(&mut self, leaves: Range<usize>) {
        let mut nodes = self.first_leaf + leaves.start..self.first_leaf + leaves.end;
        while nodes.end > 1 {
            nodes = nodes.start.saturating_sub(1) / Self::CHILDREN
                ..(nodes.end - 1).div_ceil(Self::CHILDREN);
            for node in nodes.clone().rev() {
                self.sum_children(node);
            }
        }
    }

    fn sum_children(&mut self, node: usize) {
        let scales = self.scales;
        let (parents, children) = self
            .nodes
            .split_at_mut((Self::CHILDREN * node + 1) * scales);
        let parent = &mut parents[node * scales..][..scales];
        let mut children =
            children[..children.len().min(Self::CHILDREN * scales)].chunks_exact(scales);
        parent.copy_from_slice(children.next().expect("branch has children"));
        for child in children {
            for (sum, value) in parent.iter_mut().zip(child) {
                *sum += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_candidates_reuse_their_input_allocations() {
        let (width, height) = (35, 33);
        let reference = vec![91; width * height];
        let image = GrayImage::u8(&reference, width, height, width).unwrap();
        let prepared = PreparedReference::new(image).unwrap();
        let bytes = vec![47; width * height];
        let floats = vec![0.5; width * height];
        let expected_bytes = prepared
            .incremental(GrayImage::u8(&bytes, width, height, width).unwrap())
            .unwrap()
            .score();
        let expected_floats = prepared
            .incremental(GrayImage::unit_f32(&floats, width, height, width).unwrap())
            .unwrap()
            .score();
        let byte_ptr = bytes.as_ptr();
        let float_ptr = floats.as_ptr();
        let bytes = IncrementalState::from_u8(&prepared, bytes).unwrap();
        let floats = IncrementalState::from_unit_f32(&prepared, floats).unwrap();
        let Level::Byte(byte_level) = &bytes.candidate else {
            panic!("expected a byte candidate");
        };
        let Level::Float(float_level) = &floats.candidate else {
            panic!("expected a float candidate");
        };
        assert_eq!(byte_level.samples.as_ptr(), byte_ptr);
        assert_eq!(float_level.samples.as_ptr(), float_ptr);
        assert_eq!(bytes.score(), expected_bytes);
        assert_eq!(floats.score(), expected_floats);

        assert_eq!(
            IncrementalState::from_u8(&prepared, vec![0; width * height + 1]).err(),
            Some(Error::InvalidLayout)
        );
        assert_eq!(
            IncrementalState::from_unit_f32(&prepared, vec![]).err(),
            Some(Error::InvalidLayout)
        );
        let mut invalid = vec![0.0; width * height];
        *invalid.last_mut().unwrap() = f32::INFINITY;
        assert_eq!(
            IncrementalState::from_unit_f32(&prepared, invalid).err(),
            Some(Error::NonFiniteSample)
        );
    }

    #[test]
    fn unchanged_float_update_requires_identical_sample_bits() {
        let samples = [0.0];
        let input = GrayImage::unit_f32(&samples, 1, 1, 1).unwrap();
        let prepared = PreparedReference::new(input).unwrap();
        let mut state = prepared.incremental(input).unwrap();
        let mut checkpoint = Checkpoint::default();
        let rect = Rect::new(0, 0, 1, 1);
        let negative_zero = GrayImage::unit_f32(&[-0.0], 1, 1, 1).unwrap();

        state
            .update_checkpointed(negative_zero, rect, &mut checkpoint)
            .unwrap();
        let Level::Float(candidate) = &state.candidate else {
            panic!("expected a float candidate");
        };
        assert_eq!(candidate.samples[0].to_bits(), (-0.0f32).to_bits());
        state.rollback(&checkpoint);
        let Level::Float(candidate) = &state.candidate else {
            panic!("expected a float candidate");
        };
        assert_eq!(candidate.samples[0].to_bits(), 0.0f32.to_bits());
    }
}
