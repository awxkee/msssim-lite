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
use crate::image::check_patch_bound;
use crate::pyramid::{
    self, Layout, Level, PackedLevel, PreparedPyramid, Region, TilePlane, TilePyramid, WINDOW,
    Workspace,
};
use crate::{Error, GrayImage, IncrementalState};
use std::sync::Arc;

/// Most dyadic scales a comparison can use.
pub const MAX_SCALES: usize = 6;
/// Scales used by [`PreparedReference::new`] and [`crate::compare`].
pub const DEFAULT_SCALES: usize = 3;

/// A reference image with cached scales for comparing many candidates with
/// it. Replicated edges are stored only once per cached scale. Byte references
/// keep the finest level as bytes and cache the first reduction as 16-bit sums.
pub struct PreparedReference {
    pub(crate) data: Arc<ReferenceData>,
}

/// Scratch buffers reusable across prepared-reference comparisons.
///
/// Create one per concurrent comparison. Buffers grow on demand and can be
/// reused across images and sample formats.
#[derive(Default)]
pub struct ComparisonWorkspace {
    reference: Workspace,
    candidate: Workspace,
}

impl ComparisonWorkspace {
    /// Heap bytes reserved by the scratch buffers.
    pub fn allocated_bytes(&self) -> usize {
        self.reference.allocated_bytes() + self.candidate.allocated_bytes()
    }
}

pub(crate) struct ReferenceData {
    pyramid: PreparedPyramid,
    pub(crate) layout: Layout,
    pub(crate) width: usize,
    pub(crate) height: usize,
    kernels: Kernels,
}

impl ReferenceData {
    pub(crate) fn score_tile(
        &self,
        candidate: &Level,
        region: Region,
        workspaces: &mut [Workspace; 2],
        sums: &mut [f64],
    ) {
        let [reference_scratch, candidate_scratch] = workspaces;
        let candidate = TilePyramid::from_packed(candidate, region, candidate_scratch);
        sums.fill(0.0);
        compare_tile(
            ReferenceTile::Prepared(&self.pyramid, reference_scratch),
            candidate,
            region,
            self.kernels,
            sums,
        );
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
        Self::with_cached_scales(reference, scales, scales)
    }

    /// Score at `scales` dyadic scales while retaining only the finest
    /// `cached_scales` levels, including the full-resolution image.
    /// Coarser levels are rebuilt per tile during scoring.
    ///
    /// Both counts must be positive, with `cached_scales <= scales <= MAX_SCALES`.
    pub fn with_cached_scales(
        reference: GrayImage<'_>,
        scales: usize,
        cached_scales: usize,
    ) -> Result<Self, Error> {
        let (width, height) = (reference.width(), reference.height());
        if !(1..=scales).contains(&cached_scales) {
            return Err(Error::InvalidScales);
        }
        let layout = reference_layout(width, height, scales)?;
        let (finest, bounded_floats) = reference.to_packed_with_bound(cached_scales == scales)?;
        Ok(Self::from_finest(
            finest,
            width,
            height,
            layout,
            cached_scales,
            bounded_floats,
        ))
    }

    /// Prepares a tightly packed byte reference, taking ownership of `samples`
    /// so the finest level uses the existing allocation.
    ///
    /// The vector length must equal `width * height` and `scales` must be in
    /// `1..=MAX_SCALES`. Samples are read as `value / 255`.
    pub fn from_u8(
        samples: Vec<u8>,
        width: usize,
        height: usize,
        scales: usize,
    ) -> Result<Self, Error> {
        let layout = reference_layout(width, height, scales)?;
        if samples.len() != width * height {
            return Err(Error::InvalidLayout);
        }
        let finest = Level::Byte(PackedLevel { samples, width });
        Ok(Self::from_finest(
            finest, width, height, layout, scales, false,
        ))
    }

    /// Prepares a tightly packed float reference, taking ownership of `samples`
    /// so the finest level uses the existing allocation.
    ///
    /// The vector length must equal `width * height` and `scales` must be in
    /// `1..=MAX_SCALES`. Samples must be finite and nominally in `0.0..=1.0`.
    pub fn from_unit_f32(
        samples: Vec<f32>,
        width: usize,
        height: usize,
        scales: usize,
    ) -> Result<Self, Error> {
        let layout = reference_layout(width, height, scales)?;
        if samples.len() != width * height {
            return Err(Error::InvalidLayout);
        }
        let bounded_floats = check_patch_bound(&samples)?;
        let finest = Level::Float(PackedLevel { samples, width });
        Ok(Self::from_finest(
            finest,
            width,
            height,
            layout,
            scales,
            bounded_floats,
        ))
    }

    fn from_finest(
        finest: Level,
        width: usize,
        height: usize,
        layout: Layout,
        cached_scales: usize,
        bounded_floats: bool,
    ) -> Self {
        Self {
            data: Arc::new(ReferenceData {
                pyramid: pyramid::prepare(finest, layout, cached_scales, bounded_floats),
                layout,
                width,
                height,
                kernels: Kernels::new(),
            }),
        }
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

    /// The number of levels retained by the prepared reference.
    pub fn cached_scales(&self) -> usize {
        self.data.pyramid.cached_scales()
    }

    /// Multi-scale SSIM deficit of `test` against the reference.
    pub fn compare(&self, test: GrayImage<'_>) -> Result<Score, Error> {
        self.compare_with_workspace(test, &mut ComparisonWorkspace::default())
    }

    /// Compare against `test` using caller-provided scratch buffers.
    pub fn compare_with_workspace(
        &self,
        test: GrayImage<'_>,
        workspace: &mut ComparisonWorkspace,
    ) -> Result<Score, Error> {
        let data = &self.data;
        if (test.width(), test.height()) != (data.width, data.height) {
            return Err(Error::DimensionMismatch);
        }
        compare_tiles(
            Reference::Prepared(&data.pyramid),
            test,
            data.layout,
            data.kernels,
            workspace,
        )
    }

    /// Score a candidate that matches the reference outside the patch at `(x, y)`.
    /// The patch may be strided and must fit inside the image and use the
    /// reference's sample type. Only tiles touched by the patch are evaluated.
    /// Deficits are normalized over the whole padded image at each scale.
    pub fn compare_patch(&self, patch: GrayImage<'_>, x: usize, y: usize) -> Result<Score, Error> {
        self.compare_patch_with_workspace(patch, x, y, &mut ComparisonWorkspace::default())
    }

    /// Score a patch using caller-provided scratch buffers.
    pub fn compare_patch_with_workspace(
        &self,
        patch: GrayImage<'_>,
        x: usize,
        y: usize,
        workspace: &mut ComparisonWorkspace,
    ) -> Result<Score, Error> {
        let data = &self.data;
        let (width, height) = (patch.width(), patch.height());
        if width > data.width.saturating_sub(x) || height > data.height.saturating_sub(y) {
            return Err(Error::DimensionMismatch);
        }
        let byte_input = matches!(data.pyramid.finest, Level::Byte(_));
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
        let mut sums = [0.0; MAX_SCALES];
        if data.pyramid.score_small_patch(
            pixels,
            footprint,
            data.kernels,
            &mut sums[..layout.scales],
        ) || data.pyramid.score_thin_patch(
            pixels,
            footprint,
            data.kernels,
            &mut sums[..layout.scales],
            &mut workspace.reference,
            &mut workspace.candidate,
        ) {
            return Ok(Score::from_sums(sums, layout));
        }
        for y in (y / side * side..bottom).step_by(side) {
            for x in (x / side * side..right).step_by(side) {
                let tile = Region {
                    x,
                    y,
                    width: side,
                    height: side,
                };
                let candidate = TilePyramid::with_patch(
                    &data.pyramid.finest,
                    pixels,
                    tile,
                    footprint,
                    &mut workspace.candidate,
                );
                compare_tile(
                    ReferenceTile::Prepared(&data.pyramid, &mut workspace.reference),
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

fn reference_layout(width: usize, height: usize, scales: usize) -> Result<Layout, Error> {
    if !(1..=MAX_SCALES).contains(&scales) {
        return Err(Error::InvalidScales);
    }
    Layout::new(width, height, scales)
}

pub(crate) fn compare(reference: GrayImage<'_>, test: GrayImage<'_>) -> Result<Score, Error> {
    let (width, height) = (reference.width(), reference.height());
    if (test.width(), test.height()) != (width, height) {
        return Err(Error::DimensionMismatch);
    }
    let layout = Layout::new(width, height, DEFAULT_SCALES)?;
    if reference.same_byte_view(test) {
        return Ok(Score::from_sums([0.0; MAX_SCALES], layout));
    }
    compare_tiles(
        Reference::Image(reference),
        test,
        layout,
        Kernels::new(),
        &mut ComparisonWorkspace::default(),
    )
}

enum Reference<'a> {
    Image(GrayImage<'a>),
    Prepared(&'a PreparedPyramid),
}

enum ReferenceTile<'a> {
    Image(TilePyramid<'a>),
    Prepared(&'a PreparedPyramid, &'a mut Workspace),
}

fn compare_tiles(
    reference: Reference<'_>,
    test: GrayImage<'_>,
    layout: Layout,
    kernels: Kernels,
    workspace: &mut ComparisonWorkspace,
) -> Result<Score, Error> {
    let mut deficits = [0.0; MAX_SCALES];
    for region in layout.tiles() {
        let source = match reference {
            Reference::Image(image) => {
                ReferenceTile::Image(TilePyramid::new(image, region, &mut workspace.reference)?)
            }
            Reference::Prepared(pyramid) => {
                ReferenceTile::Prepared(pyramid, &mut workspace.reference)
            }
        };
        let candidate = TilePyramid::new(test, region, &mut workspace.candidate)?;
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
        let candidate_plane = candidate.plane();
        let (tile_deficit, both_byte) = match &mut source {
            ReferenceTile::Image(source) => score_planes(kernels, source.plane(), candidate_plane),
            ReferenceTile::Prepared(pyramid, scratch) => {
                if let TilePlane::Float(candidate) = candidate_plane
                    && let Some(reference) = pyramid.interior_byte_sum(scale, region)
                {
                    (kernels.byte_sum_deficit(reference, candidate), false)
                } else {
                    score_planes(
                        kernels,
                        pyramid.tile(scale, region, scratch),
                        candidate_plane,
                    )
                }
            }
        };
        *sum += tile_deficit;
        // A zero integer-moment deficit means every byte matched. All
        // coarser windows within this aligned tile then match as well.
        if both_byte && tile_deficit == 0.0 {
            break;
        }
        if scale + 1 < scales {
            source = match source {
                ReferenceTile::Image(mut source) => {
                    source.halve(kernels);
                    ReferenceTile::Image(source)
                }
                ReferenceTile::Prepared(pyramid, scratch)
                    if scale + 1 == pyramid.cached_scales() =>
                {
                    ReferenceTile::Image(pyramid.stream_after(scale, region, scratch, kernels))
                }
                prepared => prepared,
            };
            candidate.halve(kernels);
            region = region.halve();
        }
    }
}

#[inline]
fn score_planes(kernels: Kernels, reference: TilePlane<'_>, test: TilePlane<'_>) -> (f64, bool) {
    (
        kernels.deficit(reference, test),
        matches!((reference, test), (TilePlane::Byte(_), TilePlane::Byte(_))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_self_comparison_requires_the_same_row_layout() {
        let samples: Vec<u8> = (0..12).collect();
        let packed = GrayImage::u8(&samples, 4, 2, 4).unwrap();
        let strided = GrayImage::u8(&samples, 4, 2, 8).unwrap();
        assert_eq!(compare(packed, packed).unwrap().deficit(), 0.0);
        assert!(compare(packed, strided).unwrap().deficit() > 0.0);
    }

    #[test]
    fn comparison_workspace_reuses_buffers_across_calls_and_formats() {
        let (width, height) = (41, 29);
        let reference = vec![85; width * height];
        let candidate = vec![51; width * height];
        let candidate_floats = vec![0.2; width * height];
        let prepared =
            PreparedReference::new(GrayImage::u8(&reference, width, height, width).unwrap())
                .unwrap();
        let byte_image = GrayImage::u8(&candidate, width, height, width).unwrap();
        let float_image = GrayImage::unit_f32(&candidate_floats, width, height, width).unwrap();
        let patch = GrayImage::u8(&candidate, 11, 9, width).unwrap();
        let expected_patch = prepared.compare_patch(patch, 19, 20).unwrap();
        let mut workspace = ComparisonWorkspace::default();

        for candidate in [byte_image, float_image] {
            let expected = prepared.compare(candidate).unwrap();
            assert_eq!(
                prepared.compare_with_workspace(candidate, &mut workspace),
                Ok(expected)
            );
        }
        let allocated = workspace.allocated_bytes();
        assert!(allocated > 0);
        assert_eq!(
            prepared.compare_patch_with_workspace(patch, 19, 20, &mut workspace),
            Ok(expected_patch)
        );
        assert_eq!(workspace.allocated_bytes(), allocated);
    }

    #[test]
    fn uncached_byte_levels_stream_into_reduced_scratch() {
        let (width, height) = (256, 32);
        let reference = vec![85; width * height];
        let candidate = vec![51; width * height];
        let image = GrayImage::u8(&reference, width, height, width).unwrap();
        let prepared = PreparedReference::with_cached_scales(image, 3, 2).unwrap();
        let mut workspace = ComparisonWorkspace::default();
        let candidate = GrayImage::u8(&candidate, width, height, width).unwrap();
        let score = prepared.compare_with_workspace(candidate, &mut workspace);
        assert_eq!(
            score,
            PreparedReference::new(image).unwrap().compare(candidate)
        );
        assert_eq!(workspace.reference.allocated_bytes(), 2 * 1024);
        assert_eq!(workspace.candidate.allocated_bytes(), 8 * 1024);
    }

    #[test]
    fn patch_covering_aligned_tiles_borrows_its_strided_rows() {
        let (width, height) = (32, 32);
        let reference = vec![85; width * height];
        let prepared = PreparedReference::from_u8(reference, width, height, 1).unwrap();
        let patch_stride = 19;
        let patch = vec![51; 16 * patch_stride];
        let view = GrayImage::u8(&patch, 16, 16, patch_stride).unwrap();
        let mut workspace = ComparisonWorkspace::default();
        let score = prepared
            .compare_patch_with_workspace(view, 8, 8, &mut workspace)
            .unwrap();
        let mut candidate = vec![85; width * height];
        for row in candidate.chunks_exact_mut(width).skip(8).take(16) {
            row[8..24].fill(51);
        }
        assert_eq!(
            score,
            prepared
                .compare(GrayImage::u8(&candidate, width, height, width).unwrap())
                .unwrap()
        );
        assert_eq!(workspace.allocated_bytes(), 0);
    }

    #[test]
    fn owned_references_reuse_the_input_allocation() {
        let bytes = vec![37; 17 * 13];
        let floats = vec![0.25; 17 * 13];
        let bytes_ptr = bytes.as_ptr();
        let floats_ptr = floats.as_ptr();

        let bytes = PreparedReference::from_u8(bytes, 17, 13, MAX_SCALES).unwrap();
        let floats = PreparedReference::from_unit_f32(floats, 17, 13, MAX_SCALES).unwrap();
        let Level::Byte(bytes) = &bytes.data.pyramid.finest else {
            panic!("expected a byte level");
        };
        let Level::Float(floats) = &floats.data.pyramid.finest else {
            panic!("expected a float level");
        };

        assert_eq!(bytes.samples.as_ptr(), bytes_ptr);
        assert_eq!(floats.samples.as_ptr(), floats_ptr);
    }
}
