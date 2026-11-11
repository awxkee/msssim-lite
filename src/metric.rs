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

use crate::dispatch::Kernels;
use crate::pyramid::{self, Layout, Level, Region, TilePlane, TilePyramid, WINDOW, Workspace};
use crate::{Error, GrayImage, IncrementalState};
use std::sync::Arc;

/// Most dyadic scales a comparison can use.
pub const MAX_SCALES: usize = 6;
/// Scales used by [`PreparedReference::new`] and [`crate::compare`].
pub const DEFAULT_SCALES: usize = 3;

/// A reference image padded and reduced to every scale once, for comparing
/// many candidates with it. Byte references keep their finest level as bytes.
pub struct PreparedReference {
    pub(crate) data: Arc<ReferenceData>,
}

pub(crate) struct ReferenceData {
    levels: Vec<Level>,
    pub(crate) layout: Layout,
    pub(crate) width: usize,
    pub(crate) height: usize,
    kernels: Kernels,
}

impl ReferenceData {
    pub(crate) fn score_tile(
        &self,
        image: GrayImage<'_>,
        region: Region,
        workspace: &mut Workspace,
        sums: &mut [f64],
    ) -> Result<(), Error> {
        let candidate = TilePyramid::new(image, region, workspace)?;
        sums.fill(0.0);
        compare_tile(
            ReferenceTile::Prepared(&self.levels),
            candidate,
            region,
            self.kernels,
            sums,
        );
        Ok(())
    }
}

/// The result of a comparison: per scale, the mean SSIM deficit per pixel
/// (finest first). 0 is identical; higher is worse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    deficits: [f64; MAX_SCALES],
    scales: usize,
}

impl Score {
    pub(crate) fn from_sums(mut deficits: [f64; MAX_SCALES], layout: Layout) -> Self {
        let mut windows = (layout.width / WINDOW) * (layout.height / WINDOW);
        for deficit in &mut deficits[..layout.scales] {
            *deficit /= windows as f64;
            windows /= 4;
        }
        Self {
            deficits,
            scales: layout.scales,
        }
    }

    /// Mean deficit per pixel of each scale, finest first.
    pub fn per_scale(&self) -> &[f64] {
        &self.deficits[..self.scales]
    }

    /// The scales' deficits summed (equal weights).
    pub fn deficit(&self) -> f64 {
        self.per_scale().iter().sum()
    }

    /// `1 - mean deficit over the scales`: 1 is identical.
    pub fn similarity(&self) -> f64 {
        1.0 - self.deficit() / self.scales as f64
    }
}

impl PreparedReference {
    /// Prepares `reference` at [`DEFAULT_SCALES`] scales.
    pub fn new(reference: GrayImage<'_>) -> Result<Self, Error> {
        Self::with_scales(reference, DEFAULT_SCALES)
    }

    /// Prepares `reference` at `scales` dyadic scales (`1..=MAX_SCALES`).
    /// The image is padded by edge replication so the coarsest scale is a
    /// whole number of 8x8 windows.
    pub fn with_scales(reference: GrayImage<'_>, scales: usize) -> Result<Self, Error> {
        if !(1..=MAX_SCALES).contains(&scales) {
            return Err(Error::InvalidScales);
        }
        let (width, height) = (reference.width(), reference.height());
        let layout = Layout::new(width, height, scales)?;
        let kernels = Kernels::new();
        Ok(Self {
            data: Arc::new(ReferenceData {
                levels: pyramid::prepare(reference, layout, kernels)?,
                layout,
                width,
                height,
                kernels,
            }),
        })
    }

    pub fn width(&self) -> usize {
        self.data.width
    }

    pub fn height(&self) -> usize {
        self.data.height
    }

    pub fn scales(&self) -> usize {
        self.data.layout.scales
    }

    /// Multi-scale SSIM deficit of `test` against the reference.
    pub fn compare(&self, test: GrayImage<'_>) -> Result<Score, Error> {
        let data = &self.data;
        if (test.width(), test.height()) != (data.width, data.height) {
            return Err(Error::DimensionMismatch);
        }
        compare_tiles(
            Reference::Prepared(&data.levels),
            test,
            data.layout,
            data.kernels,
        )
    }

    /// Score a candidate that matches the reference outside the patch at `(x, y)`.
    /// The patch may be strided and must fit inside the image and use the
    /// reference's sample type. Only tiles touched by the patch are evaluated.
    /// Deficits are normalized over the whole padded image at each scale.
    pub fn compare_patch(&self, patch: GrayImage<'_>, x: usize, y: usize) -> Result<Score, Error> {
        let data = &self.data;
        let (width, height) = (patch.width(), patch.height());
        if width > data.width.saturating_sub(x) || height > data.height.saturating_sub(y) {
            return Err(Error::DimensionMismatch);
        }
        let byte_input = matches!(data.levels[0], Level::Byte(_));
        if patch.is_byte() != byte_input {
            return Err(Error::InvalidLayout);
        }
        let pixels = patch.checked_region(Region {
            x: 0,
            y: 0,
            width,
            height,
        })?;

        let layout = data.layout;
        let side = layout.alignment();
        // A patch touching the right or bottom edge changes its replicas too.
        let right = if x + width == data.width {
            layout.width
        } else {
            x + width
        };
        let bottom = if y + height == data.height {
            layout.height
        } else {
            y + height
        };
        let footprint = Region {
            x,
            y,
            width: right - x,
            height: bottom - y,
        };
        let mut workspace = Workspace::with_area(side * side, layout.scales, byte_input);
        let mut sums = [0.0; MAX_SCALES];
        for y in (y / side * side..bottom).step_by(side) {
            for x in (x / side * side..right).step_by(side) {
                let tile = Region {
                    x,
                    y,
                    width: side,
                    height: side,
                };
                let candidate = TilePyramid::with_patch(
                    data.levels[0].region(tile),
                    pixels,
                    tile,
                    footprint,
                    &mut workspace,
                );
                compare_tile(
                    ReferenceTile::Prepared(&data.levels),
                    candidate,
                    tile,
                    data.kernels,
                    &mut sums[..layout.scales],
                );
            }
        }
        Ok(Score::from_sums(sums, layout))
    }

    /// Score `test` and retain the state needed for local updates.
    pub fn incremental(&self, test: GrayImage<'_>) -> Result<IncrementalState, Error> {
        IncrementalState::new(self, test)
    }
}

pub(crate) fn compare(reference: GrayImage<'_>, test: GrayImage<'_>) -> Result<Score, Error> {
    let (width, height) = (reference.width(), reference.height());
    if (test.width(), test.height()) != (width, height) {
        return Err(Error::DimensionMismatch);
    }
    let layout = Layout::new(width, height, DEFAULT_SCALES)?;
    compare_tiles(Reference::Image(reference), test, layout, Kernels::new())
}

enum Reference<'a> {
    Image(GrayImage<'a>),
    Prepared(&'a [Level]),
}

enum ReferenceTile<'a> {
    Image(TilePyramid<'a>),
    Prepared(&'a [Level]),
}

fn compare_tiles(
    reference: Reference<'_>,
    test: GrayImage<'_>,
    layout: Layout,
    kernels: Kernels,
) -> Result<Score, Error> {
    let mut reference_scratch = match reference {
        Reference::Image(image) => Workspace::new(layout, image.is_byte()),
        Reference::Prepared(_) => Workspace::default(),
    };
    let mut test_scratch = Workspace::new(layout, test.is_byte());
    let mut deficits = [0.0; MAX_SCALES];
    for region in layout.tiles() {
        let source = match reference {
            Reference::Image(image) => {
                ReferenceTile::Image(TilePyramid::new(image, region, &mut reference_scratch)?)
            }
            Reference::Prepared(levels) => ReferenceTile::Prepared(levels),
        };
        let candidate = TilePyramid::new(test, region, &mut test_scratch)?;
        compare_tile(
            source,
            candidate,
            region,
            kernels,
            &mut deficits[..layout.scales],
        );
    }
    Ok(Score::from_sums(deficits, layout))
}

#[inline]
fn compare_tile(
    mut source: ReferenceTile<'_>,
    mut candidate: TilePyramid<'_>,
    mut region: Region,
    kernels: Kernels,
    sums: &mut [f64],
) {
    let scales = sums.len();
    for (scale, sum) in sums.iter_mut().enumerate() {
        let reference_plane = match &source {
            ReferenceTile::Image(source) => source.plane(),
            ReferenceTile::Prepared(levels) => levels[scale].region(region),
        };
        let candidate_plane = candidate.plane();
        let tile_deficit = kernels.deficit(reference_plane, candidate_plane);
        *sum += tile_deficit;
        // A zero integer-moment deficit means every byte matched. All
        // coarser windows within this aligned tile then match as well.
        if matches!(
            (reference_plane, candidate_plane),
            (TilePlane::Byte(_), TilePlane::Byte(_))
        ) && tile_deficit == 0.0
        {
            break;
        }
        if scale + 1 < scales {
            if let ReferenceTile::Image(source) = &mut source {
                source.halve(kernels);
            }
            candidate.halve(kernels);
            region = region.halve();
        }
    }
}
