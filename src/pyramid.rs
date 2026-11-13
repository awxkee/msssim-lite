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

//! Aligned tiles and 2×2 box reduction for the dyadic pyramid.

use crate::dispatch::Kernels;
use crate::image::within_patch_bound;
use crate::{Error, GrayImage};

pub(crate) const WINDOW: usize = 8;
const TILE_WIDTH: usize = 256;
const MIN_TILE_HEIGHT: usize = 32;
const _: () = assert!(TILE_WIDTH.is_multiple_of(WINDOW << (crate::MAX_SCALES - 1)));

#[derive(Clone, Copy)]
pub(crate) struct Plane<'a, T = f32> {
    pub(crate) samples: &'a [T],
    pub(crate) stride: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<'a, T> Plane<'a, T> {
    pub(crate) fn packed(samples: &'a [T], region: Region) -> Self {
        Self {
            samples: &samples[..region.width * region.height],
            stride: region.width,
            width: region.width,
            height: region.height,
        }
    }

    pub(crate) fn subregion(self, region: Region) -> Self {
        let start = region.y * self.stride + region.x;
        let len = (region.height - 1) * self.stride + region.width;
        Self {
            samples: &self.samples[start..start + len],
            stride: self.stride,
            width: region.width,
            height: region.height,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum TilePlane<'a> {
    Byte(Plane<'a, u8>),
    Float(Plane<'a>),
}

#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Region {
    pub(crate) fn halve(self) -> Self {
        Self {
            x: self.x / 2,
            y: self.y / 2,
            width: self.width / 2,
            height: self.height / 2,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) scales: usize,
}

impl Layout {
    pub(crate) fn new(width: usize, height: usize, scales: usize) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions);
        }
        let align = WINDOW << (scales - 1);
        let (width, height) = width
            .checked_next_multiple_of(align)
            .zip(height.checked_next_multiple_of(align))
            .filter(|&(w, h)| w <= (isize::MAX as usize / size_of::<f32>()) / h)
            .ok_or(Error::InvalidDimensions)?;
        Ok(Self {
            width,
            height,
            scales,
        })
    }

    fn tile_height(self) -> usize {
        MIN_TILE_HEIGHT.max(self.alignment())
    }

    pub(crate) fn alignment(self) -> usize {
        WINDOW << (self.scales - 1)
    }

    pub(crate) fn tiles(self) -> impl Iterator<Item = Region> {
        // Align tile boundaries to whole windows at the coarsest scale.
        let tile_height = self.tile_height();
        (0..self.height).step_by(tile_height).flat_map(move |y| {
            (0..self.width).step_by(TILE_WIDTH).map(move |x| Region {
                x,
                y,
                width: TILE_WIDTH.min(self.width - x),
                height: tile_height.min(self.height - y),
            })
        })
    }
}

pub(crate) enum Level {
    Byte(PackedLevel<u8>),
    Float(PackedLevel<f32>),
}

impl Level {
    pub(crate) fn region(&self, region: Region) -> TilePlane<'_> {
        match self {
            Self::Byte(level) => TilePlane::Byte(level.region(region)),
            Self::Float(level) => TilePlane::Float(level.region(region)),
        }
    }

    fn tile<'a>(&'a self, region: Region, scratch: &'a mut Workspace) -> TilePlane<'a> {
        match self {
            Self::Byte(level) => TilePlane::Byte(level.tile(region, &mut scratch.bytes)),
            Self::Float(level) => TilePlane::Float(level.tile(region, &mut scratch.floats)),
        }
    }
}

pub(crate) struct PackedLevel<T> {
    pub(crate) samples: Vec<T>,
    pub(crate) width: usize,
}

impl<T> PackedLevel<T> {
    fn height(&self) -> usize {
        self.samples.len() / self.width
    }

    fn interior(&self, region: Region) -> Option<Plane<'_, T>> {
        (region.x + region.width <= self.width
            && (region.y + region.height) * self.width <= self.samples.len())
        .then(|| self.region(region))
    }

    fn region(&self, region: Region) -> Plane<'_, T> {
        let start = region.y * self.width + region.x;
        let len = (region.height - 1) * self.width + region.width;
        Plane {
            samples: &self.samples[start..start + len],
            stride: self.width,
            width: region.width,
            height: region.height,
        }
    }
}

impl<T: Copy + Default> PackedLevel<T> {
    fn tile<'a>(&'a self, region: Region, scratch: &'a mut Vec<T>) -> Plane<'a, T> {
        if let Some(plane) = self.interior(region) {
            return plane;
        }
        let output = scratch_slice(scratch, region.width * region.height);
        self.write_tile(region, output, |src, dst| dst.copy_from_slice(src));
        Plane::packed(output, region)
    }

    fn write_tile<U: Copy>(&self, region: Region, output: &mut [U], copy: impl Fn(&[T], &mut [U])) {
        let visible = Region {
            width: region.width.min(self.width - region.x),
            height: region.height.min(self.height() - region.y),
            ..region
        };
        let source = self.region(visible);
        let (rows, bottom) = output.split_at_mut(visible.height * region.width);
        for (src, dst) in source
            .samples
            .chunks(source.stride)
            .zip(rows.chunks_exact_mut(region.width))
        {
            let (pixels, right) = dst.split_at_mut(visible.width);
            copy(&src[..visible.width], pixels);
            right.fill(pixels[visible.width - 1]);
        }
        let last = &rows[rows.len() - region.width..];
        for dst in bottom.chunks_exact_mut(region.width) {
            dst.copy_from_slice(last);
        }
    }
}

pub(crate) struct PreparedPyramid {
    pub(crate) finest: Level,
    reduced: Vec<ReducedLevel>,
    bounded_floats: bool,
}

enum ReducedLevel {
    // A 2×2 byte sum fits in 10 bits. Converting it with the original
    // multiplier reproduces the floating-point sample exactly.
    ByteSums(PackedLevel<u16>),
    Float(PackedLevel<f32>),
}

impl PreparedPyramid {
    pub(crate) fn cached_scales(&self) -> usize {
        1 + self.reduced.len()
    }

    pub(crate) fn interior_byte_sum(&self, scale: usize, region: Region) -> Option<Plane<'_, u16>> {
        let index = scale.checked_sub(1)?;
        let ReducedLevel::ByteSums(level) = self.reduced.get(index)? else {
            return None;
        };
        level.interior(region)
    }

    pub(crate) fn tile<'a>(
        &'a self,
        scale: usize,
        region: Region,
        scratch: &'a mut Workspace,
    ) -> TilePlane<'a> {
        if scale == 0 {
            return self.finest.tile(region, scratch);
        }
        TilePlane::Float(match &self.reduced[scale - 1] {
            ReducedLevel::ByteSums(level) => {
                let output = scratch_slice(&mut scratch.floats, region.width * region.height);
                decode_byte_sums(level, region, output);
                Plane::packed(output, region)
            }
            ReducedLevel::Float(level) => level.tile(region, &mut scratch.floats),
        })
    }

    pub(crate) fn stream_after<'a>(
        &'a self,
        scale: usize,
        region: Region,
        scratch: &'a mut Workspace,
        kernels: Kernels,
    ) -> TilePyramid<'a> {
        if scale == 0 {
            let mut tile = TilePyramid::from_packed(&self.finest, region, scratch);
            tile.halve(kernels);
            return tile;
        }
        let borrowed = match &self.reduced[scale - 1] {
            ReducedLevel::ByteSums(level) => {
                if let Some(input) = level.interior(region) {
                    let next = region.halve();
                    let output = scratch_slice(&mut scratch.floats, next.width * next.height);
                    // Apply the same float conversion to each sum before
                    // averaging, without materializing the decoded level.
                    halve(input, output, average_byte_sums);
                    return TilePyramid {
                        borrowed: None,
                        floats: &mut scratch.floats,
                        region: next,
                    };
                }
                let tile = scratch_slice(&mut scratch.floats, region.width * region.height);
                decode_byte_sums(level, region, tile);
                None
            }
            ReducedLevel::Float(level) => {
                TilePyramid::load_float(level, region, &mut scratch.floats)
            }
        };
        let mut tile = TilePyramid {
            borrowed,
            floats: &mut scratch.floats,
            region,
        };
        tile.halve(kernels);
        tile
    }
}

impl PreparedPyramid {
    pub(crate) fn score_small_patch(
        &self,
        patch: TilePlane<'_>,
        footprint: Region,
        kernels: Kernels,
        sums: &mut [f64],
    ) -> bool {
        const HALF_WINDOW: usize = WINDOW / 2;
        if self.cached_scales() != sums.len() {
            return false;
        }
        if footprint.x % WINDOW + footprint.width > WINDOW
            || footprint.y % WINDOW + footprint.height > WINDOW
        {
            return self.score_multi_window_patch(patch, footprint, kernels, sums);
        }

        let mut window = Region {
            x: footprint.x / WINDOW * WINDOW,
            y: footprint.y / WINDOW * WINDOW,
            width: WINDOW,
            height: WINDOW,
        };
        let (finest_sum, reduced_sums) = sums.split_first_mut().expect("finest scale");
        let mut reduced = [0.0; HALF_WINDOW * HALF_WINDOW];
        match (&self.finest, patch) {
            (Level::Byte(finest), TilePlane::Byte(patch)) => {
                let (reference, candidate) = patched_window(finest, patch, window, footprint);
                let candidate = Plane::packed(&candidate, window);
                let deficit = kernels.deficit(
                    TilePlane::Byte(Plane::packed(&reference, window)),
                    TilePlane::Byte(candidate),
                );
                *finest_sum += deficit;
                if deficit == 0.0 || reduced_sums.is_empty() {
                    return true;
                }
                halve(candidate, &mut reduced, average_byte);
            }
            (Level::Float(finest), TilePlane::Float(patch))
                if self.bounded_floats && patch_is_bounded(patch) =>
            {
                let (reference, candidate) = patched_window(finest, patch, window, footprint);
                let candidate = Plane::packed(&candidate, window);
                *finest_sum += kernels.deficit(
                    TilePlane::Float(Plane::packed(&reference, window)),
                    TilePlane::Float(candidate),
                );
                if reduced_sums.is_empty() {
                    return true;
                }
                halve(candidate, &mut reduced, average_float);
            }
            _ => return false,
        }

        let mut reference = [0.0; WINDOW * WINDOW];
        for (index, (level, sum)) in self.reduced.iter().zip(reduced_sums).enumerate() {
            // Each changed window maps to one 4×4 quadrant of its parent.
            // Every other window still matches the reference exactly.
            let quadrant = window.halve();
            let (x, y) = (quadrant.x % WINDOW, quadrant.y % WINDOW);
            window.x = quadrant.x - x;
            window.y = quadrant.y - y;
            // The parent contains the reduced position of the edited pixel,
            // so its origin still lies inside the stored extent.
            match level {
                ReducedLevel::ByteSums(level) => decode_byte_sums(level, window, &mut reference),
                ReducedLevel::Float(level) => {
                    level.write_tile(window, &mut reference, |src, dst| dst.copy_from_slice(src));
                }
            }
            let mut candidate = reference;
            for (row, values) in reduced.as_chunks::<HALF_WINDOW>().0.iter().enumerate() {
                let start = (y + row) * WINDOW + x;
                candidate[start..start + HALF_WINDOW].copy_from_slice(values);
            }
            let candidate = Plane::packed(&candidate, window);
            *sum += kernels.deficit(
                TilePlane::Float(Plane::packed(&reference, window)),
                TilePlane::Float(candidate),
            );
            if index + 1 < self.reduced.len() {
                halve(candidate, &mut reduced, average_float);
            }
        }
        true
    }

    fn score_multi_window_patch(
        &self,
        patch: TilePlane<'_>,
        footprint: Region,
        kernels: Kernels,
        sums: &mut [f64],
    ) -> bool {
        let tile_side = WINDOW << (sums.len() - 1);
        // At low scale counts there is little work to skip. For larger
        // tiles only use this path when the affected region is a fraction
        // of the full tile and fits the stack buffers.
        let limit = (tile_side / 2).min(SMALL_PATCH_SIDE);
        let windows = enclosing_windows(footprint);
        if windows.width > limit || windows.height > limit {
            return false;
        }
        if let TilePlane::Float(patch) = patch
            && !(self.bounded_floats && patch_is_bounded(patch))
        {
            return false;
        }

        let right = windows.x + windows.width;
        let bottom = windows.y + windows.height;
        for y in (windows.y / tile_side * tile_side..bottom).step_by(tile_side) {
            for x in (windows.x / tile_side * tile_side..right).step_by(tile_side) {
                // Keep the same tile boundaries and raster order as
                // compare_patch so each f64 subtotal is rounded as before.
                let left = x.max(windows.x);
                let top = y.max(windows.y);
                self.score_patch_windows(
                    patch,
                    footprint,
                    Region {
                        x: left,
                        y: top,
                        width: (x + tile_side).min(right) - left,
                        height: (y + tile_side).min(bottom) - top,
                    },
                    kernels,
                    sums,
                );
            }
        }
        true
    }

    #[inline(never)]
    fn score_patch_windows(
        &self,
        patch: TilePlane<'_>,
        footprint: Region,
        windows: Region,
        kernels: Kernels,
        sums: &mut [f64],
    ) {
        let (finest_sum, reduced_sums) = sums.split_first_mut().expect("finest scale");
        const AREA: usize = SMALL_PATCH_SIDE * SMALL_PATCH_SIDE;
        let mut reduced = [0.0; AREA / 4];
        let mut reference = [0.0; AREA];
        let mut candidate = [0.0; AREA];
        let area = windows.width * windows.height;

        match (&self.finest, patch) {
            (Level::Byte(finest), TilePlane::Byte(patch)) => {
                let mut reference = [0; AREA];
                let mut candidate = [0; AREA];
                let reference = &mut reference[..area];
                let candidate = &mut candidate[..area];
                write_patched_window(finest, patch, windows, footprint, reference, candidate);
                let candidate = Plane::packed(candidate, windows);
                let deficit = kernels.deficit(
                    TilePlane::Byte(Plane::packed(reference, windows)),
                    TilePlane::Byte(candidate),
                );
                *finest_sum += deficit;
                if deficit == 0.0 || reduced_sums.is_empty() {
                    return;
                }
                halve(candidate, &mut reduced[..area / 4], average_byte);
            }
            (Level::Float(finest), TilePlane::Float(patch)) => {
                write_patched_window(
                    finest,
                    patch,
                    windows,
                    footprint,
                    &mut reference[..area],
                    &mut candidate[..area],
                );
                let candidate = Plane::packed(&candidate, windows);
                *finest_sum += kernels.deficit(
                    TilePlane::Float(Plane::packed(&reference, windows)),
                    TilePlane::Float(candidate),
                );
                if reduced_sums.is_empty() {
                    return;
                }
                halve(candidate, &mut reduced[..area / 4], average_float);
            }
            _ => unreachable!("patch format was checked"),
        }

        self.score_patch_float_levels(
            windows,
            kernels,
            reduced_sums,
            &mut reference,
            &mut candidate,
            &mut reduced,
        );
    }

    fn score_patch_float_levels(
        &self,
        mut windows: Region,
        kernels: Kernels,
        sums: &mut [f64],
        reference: &mut [f32],
        candidate: &mut [f32],
        reduced: &mut [f32],
    ) {
        for (index, (level, sum)) in self.reduced.iter().zip(sums).enumerate() {
            let changed = windows.halve();
            windows = enclosing_windows(changed);
            let area = windows.width * windows.height;
            let reference = &mut reference[..area];
            let candidate = &mut candidate[..area];
            match level {
                ReducedLevel::ByteSums(level) => decode_byte_sums(level, windows, reference),
                ReducedLevel::Float(level) => {
                    level.write_tile(windows, reference, |src, dst| dst.copy_from_slice(src));
                }
            }
            candidate.copy_from_slice(reference);
            for (row, src) in reduced[..changed.width * changed.height]
                .chunks_exact(changed.width)
                .enumerate()
            {
                let start = (changed.y - windows.y + row) * windows.width + changed.x - windows.x;
                candidate[start..start + changed.width].copy_from_slice(src);
            }
            let candidate = Plane::packed(candidate, windows);
            *sum += kernels.deficit(
                TilePlane::Float(Plane::packed(reference, windows)),
                TilePlane::Float(candidate),
            );
            if index + 1 < self.reduced.len() {
                halve(candidate, &mut reduced[..area / 4], average_float);
            }
        }
    }

    pub(crate) fn score_thin_patch(
        &self,
        patch: TilePlane<'_>,
        footprint: Region,
        kernels: Kernels,
        sums: &mut [f64],
        reference: &mut Workspace,
        candidate: &mut Workspace,
    ) -> bool {
        let tile_side = WINDOW << (sums.len() - 1);
        if tile_side <= SMALL_PATCH_SIDE || self.cached_scales() != sums.len() {
            return false;
        }
        let mut windows = enclosing_windows(footprint);
        if matches!(patch, TilePlane::Byte(_)) {
            // The AVX2 byte kernel accumulates neighboring windows as pairs.
            // Keep those pairs aligned to the original scoring tile.
            let x = windows.x / (2 * WINDOW) * (2 * WINDOW);
            windows.width = (windows.x + windows.width).next_multiple_of(2 * WINDOW) - x;
            windows.x = x;
        }
        let max_area = windows.width.min(tile_side) * windows.height.min(tile_side);
        if max_area > tile_side * tile_side / 4 {
            return false;
        }
        if let TilePlane::Float(patch) = patch
            && !(self.bounded_floats && patch_is_bounded(patch))
        {
            return false;
        }

        let right = windows.x + windows.width;
        let bottom = windows.y + windows.height;
        for y in (windows.y / tile_side * tile_side..bottom).step_by(tile_side) {
            for x in (windows.x / tile_side * tile_side..right).step_by(tile_side) {
                let left = x.max(windows.x);
                let top = y.max(windows.y);
                self.score_thin_patch_tile(
                    patch,
                    footprint,
                    Region {
                        x: left,
                        y: top,
                        width: (x + tile_side).min(right) - left,
                        height: (y + tile_side).min(bottom) - top,
                    },
                    kernels,
                    sums,
                    (reference, candidate),
                );
            }
        }
        true
    }

    fn score_thin_patch_tile(
        &self,
        patch: TilePlane<'_>,
        footprint: Region,
        windows: Region,
        kernels: Kernels,
        sums: &mut [f64],
        (reference, candidate): (&mut Workspace, &mut Workspace),
    ) {
        let area = windows.width * windows.height;
        let coarse = enclosing_windows(windows.halve());
        let float_area = match patch {
            TilePlane::Byte(_) => coarse.width * coarse.height,
            TilePlane::Float(_) => area,
        };
        let (reference_float, reduced) =
            scratch_slice(&mut reference.floats, float_area + area / 4).split_at_mut(float_area);
        let candidate_float = scratch_slice(&mut candidate.floats, float_area);
        let (finest_sum, reduced_sums) = sums.split_first_mut().expect("finest scale");

        match (&self.finest, patch) {
            (Level::Byte(finest), TilePlane::Byte(patch)) => {
                let reference = finest.tile(windows, &mut reference.bytes);
                let candidate = scratch_slice(&mut candidate.bytes, area);
                copy_plane(reference, candidate);
                overlay_patch(patch, windows, footprint, candidate);
                let candidate = Plane::packed(candidate, windows);
                let deficit =
                    kernels.deficit(TilePlane::Byte(reference), TilePlane::Byte(candidate));
                *finest_sum += deficit;
                if deficit == 0.0 {
                    return;
                }
                kernels.halve_bytes(candidate, reduced);
            }
            (Level::Float(finest), TilePlane::Float(patch)) => {
                let reference = if let Some(plane) = finest.interior(windows) {
                    plane
                } else {
                    finest.write_tile(windows, reference_float, |src, dst| {
                        dst.copy_from_slice(src)
                    });
                    Plane::packed(reference_float, windows)
                };
                copy_plane(reference, candidate_float);
                overlay_patch(patch, windows, footprint, candidate_float);
                let candidate = Plane::packed(candidate_float, windows);
                *finest_sum +=
                    kernels.deficit(TilePlane::Float(reference), TilePlane::Float(candidate));
                halve(candidate, reduced, average_float);
            }
            _ => unreachable!("patch format was checked"),
        }
        self.score_patch_float_levels(
            windows,
            kernels,
            reduced_sums,
            reference_float,
            candidate_float,
            reduced,
        );
    }
}

const SMALL_PATCH_SIDE: usize = 4 * WINDOW;

fn enclosing_windows(region: Region) -> Region {
    let x = region.x / WINDOW * WINDOW;
    let y = region.y / WINDOW * WINDOW;
    Region {
        x,
        y,
        width: (region.x + region.width).next_multiple_of(WINDOW) - x,
        height: (region.y + region.height).next_multiple_of(WINDOW) - y,
    }
}

fn patch_is_bounded(patch: Plane<'_>) -> bool {
    patch
        .samples
        .chunks(patch.stride)
        .all(|row| within_patch_bound(&row[..patch.width]))
}

fn patched_window<T: Copy + Default>(
    finest: &PackedLevel<T>,
    patch: Plane<'_, T>,
    window: Region,
    footprint: Region,
) -> ([T; WINDOW * WINDOW], [T; WINDOW * WINDOW]) {
    let mut reference = [T::default(); WINDOW * WINDOW];
    let mut candidate = reference;
    write_patched_window(
        finest,
        patch,
        window,
        footprint,
        &mut reference,
        &mut candidate,
    );
    (reference, candidate)
}

fn write_patched_window<T: Copy + Default>(
    finest: &PackedLevel<T>,
    patch: Plane<'_, T>,
    window: Region,
    footprint: Region,
    reference: &mut [T],
    candidate: &mut [T],
) {
    finest.write_tile(window, reference, |src, dst| dst.copy_from_slice(src));
    candidate.copy_from_slice(reference);
    overlay_patch(patch, window, footprint, candidate);
}

fn copy_plane<T: Copy>(source: Plane<'_, T>, output: &mut [T]) {
    for (src, dst) in source
        .samples
        .chunks(source.stride)
        .zip(output.chunks_exact_mut(source.width))
    {
        dst.copy_from_slice(&src[..source.width]);
    }
}

fn decode_byte_sums(level: &PackedLevel<u16>, region: Region, output: &mut [f32]) {
    level.write_tile(region, output, |src, dst| {
        for (sum, pixel) in src.iter().zip(dst) {
            *pixel = normalize_sum(*sum);
        }
    });
}

pub(crate) fn prepare(
    finest: Level,
    layout: Layout,
    cached_scales: usize,
    bounded_floats: bool,
) -> PreparedPyramid {
    let mut reduced = Vec::with_capacity(cached_scales - 1);
    let mut region = Region {
        x: 0,
        y: 0,
        width: layout.width,
        height: layout.height,
    };
    for _ in 1..cached_scales {
        region = region.halve();
        let level = match reduced.last() {
            Some(ReducedLevel::ByteSums(input)) => {
                ReducedLevel::Float(reduce_level(input, region, average_byte_sums))
            }
            Some(ReducedLevel::Float(input)) => {
                ReducedLevel::Float(reduce_level(input, region, average_float))
            }
            None => match &finest {
                Level::Byte(input) => ReducedLevel::ByteSums(reduce_level(input, region, byte_sum)),
                Level::Float(input) => {
                    ReducedLevel::Float(reduce_level(input, region, average_float))
                }
            },
        };
        reduced.push(level);
    }
    PreparedPyramid {
        finest,
        reduced,
        bounded_floats,
    }
}

fn reduce_level<T: Copy, U>(
    input: &PackedLevel<T>,
    padded: Region,
    average: impl Fn(&[T; 2], &[T; 2]) -> U + Copy,
) -> PackedLevel<U> {
    let input_height = input.height();
    // Keep one sample whose entire filter footprint is at the replicated
    // edge. A preceding sample can still mix real and replicated pixels.
    let width = (input.width / 2 + 1).min(padded.width);
    let height = (input_height / 2 + 1).min(padded.height);
    let mut samples = Vec::with_capacity(width * height);
    let row = |y: usize| {
        let start = y.min(input_height - 1) * input.width;
        &input.samples[start..start + input.width]
    };
    for y in 0..height {
        let upper = row(2 * y);
        let lower = row(2 * y + 1);
        samples.extend(reduce_row(upper, lower, average));
        if width > input.width / 2 {
            samples.push(average(
                &[upper[input.width - 1]; 2],
                &[lower[input.width - 1]; 2],
            ));
        }
    }
    PackedLevel { samples, width }
}

#[derive(Default)]
pub(crate) struct Workspace {
    floats: Vec<f32>,
    bytes: Vec<u8>,
}

impl Workspace {
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.floats.capacity() * size_of::<f32>() + self.bytes.capacity()
    }
}

pub(crate) fn scratch_slice<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) -> &mut [T] {
    if len > buffer.len() {
        buffer.reserve_exact(len - buffer.len());
        buffer.resize(len, T::default());
    }
    &mut buffer[..len]
}

/// A tile's current level. Once owned, float levels shrink in place.
pub(crate) struct TilePyramid<'a> {
    borrowed: Option<TilePlane<'a>>,
    floats: &'a mut Vec<f32>,
    region: Region,
}

impl<'a> TilePyramid<'a> {
    pub(crate) fn new(
        image: GrayImage<'a>,
        region: Region,
        scratch: &'a mut Workspace,
    ) -> Result<Self, Error> {
        let borrowed = image.borrow_region(region, &mut scratch.bytes)?;
        if borrowed.is_none() {
            let tile = scratch_slice(&mut scratch.floats, region.width * region.height);
            image.write_float_region(region, tile)?;
        }
        Ok(Self {
            borrowed,
            floats: &mut scratch.floats,
            region,
        })
    }

    /// Load a tile from the incremental state's validated candidate samples.
    pub(crate) fn from_packed(
        level: &'a Level,
        region: Region,
        scratch: &'a mut Workspace,
    ) -> Self {
        let borrowed = match level {
            Level::Byte(level) => Some(TilePlane::Byte(level.tile(region, &mut scratch.bytes))),
            Level::Float(level) => Self::load_float(level, region, &mut scratch.floats),
        };
        Self {
            borrowed,
            floats: &mut scratch.floats,
            region,
        }
    }

    fn load_float(
        level: &'a PackedLevel<f32>,
        region: Region,
        scratch: &mut Vec<f32>,
    ) -> Option<TilePlane<'a>> {
        if let Some(plane) = level.interior(region) {
            return Some(TilePlane::Float(plane));
        }
        let tile = scratch_slice(scratch, region.width * region.height);
        level.write_tile(region, tile, |src, dst| dst.copy_from_slice(src));
        None
    }

    pub(crate) fn with_patch(
        reference: &Level,
        patch: TilePlane<'a>,
        region: Region,
        footprint: Region,
        scratch: &'a mut Workspace,
    ) -> Self {
        if region.x >= footprint.x && region.y >= footprint.y {
            let offset = Region {
                x: region.x - footprint.x,
                y: region.y - footprint.y,
                ..region
            };
            let (width, height) = match patch {
                TilePlane::Byte(plane) => (plane.width, plane.height),
                TilePlane::Float(plane) => (plane.width, plane.height),
            };
            if offset.x + offset.width <= width && offset.y + offset.height <= height {
                let borrowed = match patch {
                    TilePlane::Byte(plane) => TilePlane::Byte(plane.subregion(offset)),
                    TilePlane::Float(plane) => TilePlane::Float(plane.subregion(offset)),
                };
                return Self {
                    borrowed: Some(borrowed),
                    floats: &mut scratch.floats,
                    region,
                };
            }
        }
        let area = region.width * region.height;
        let borrowed = match (reference, patch) {
            (Level::Byte(reference), TilePlane::Byte(patch)) => {
                let tile = scratch_slice(&mut scratch.bytes, area);
                reference.write_tile(region, tile, |src, dst| dst.copy_from_slice(src));
                overlay_patch(patch, region, footprint, tile);
                Some(TilePlane::Byte(Plane::packed(tile, region)))
            }
            (Level::Float(reference), TilePlane::Float(patch)) => {
                let tile = scratch_slice(&mut scratch.floats, area);
                reference.write_tile(region, tile, |src, dst| dst.copy_from_slice(src));
                overlay_patch(patch, region, footprint, tile);
                None
            }
            _ => unreachable!("patch format was checked"),
        };
        Self {
            borrowed,
            floats: &mut scratch.floats,
            region,
        }
    }

    pub(crate) fn plane(&self) -> TilePlane<'_> {
        self.borrowed
            .unwrap_or_else(|| TilePlane::Float(Plane::packed(self.floats, self.region)))
    }

    pub(crate) fn halve(&mut self, kernels: Kernels) {
        let fine = self.region;
        self.region = fine.halve();
        if let Some(input) = self.borrowed.take() {
            let output = scratch_slice(self.floats, self.region.width * self.region.height);
            match input {
                TilePlane::Byte(input) => kernels.halve_bytes(input, output),
                TilePlane::Float(input) => halve(input, output, average_float),
            }
        } else {
            kernels.halve_float_in_place(self.floats, fine.width, fine.height);
        }
    }
}

fn overlay_patch<T: Copy>(patch: Plane<'_, T>, tile: Region, footprint: Region, output: &mut [T]) {
    let left = tile.x.max(footprint.x);
    let right = (tile.x + tile.width).min(footprint.x + footprint.width);
    let patch_x = left - footprint.x;
    // The final aligned tile contains both the edge pixel and its replicas.
    let visible = (right - left).min(patch.width - patch_x);
    for (row, dst) in output.chunks_exact_mut(tile.width).enumerate() {
        let y = tile.y + row;
        if (footprint.y..footprint.y + footprint.height).contains(&y) {
            let patch_row = (y - footprint.y).min(patch.height - 1) * patch.stride;
            let (pixels, padding) = dst[left - tile.x..right - tile.x].split_at_mut(visible);
            pixels.copy_from_slice(&patch.samples[patch_row + patch_x..][..visible]);
            padding.fill(patch.samples[patch_row + patch.width - 1]);
        }
    }
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn halve_bytes(input: Plane<'_, u8>, output: &mut [f32]) {
    halve(input, output, average_byte);
}

fn halve<T>(
    input: Plane<'_, T>,
    output: &mut [f32],
    average: impl Fn(&[T; 2], &[T; 2]) -> f32 + Copy,
) {
    for (dst, src) in output
        .chunks_exact_mut(input.width / 2)
        .zip(reduced_rows(input, average))
    {
        for (dst, value) in dst.iter_mut().zip(src) {
            *dst = value;
        }
    }
}

#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "avx", target_arch = "x86_64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[inline(always)]
pub(crate) fn halve_float_blocks(
    samples: &mut [f32],
    width: usize,
    height: usize,
    mut reduce: impl FnMut(&[f32; 16], &[f32; 16]) -> [f32; 8],
) {
    assert!(width >= 16 && width.is_multiple_of(16));
    let half_width = width / 2;

    // Row zero shares storage with its upper input row. Read both input
    // blocks before writing the result back over samples already read.
    let (upper, lower) = samples[..2 * width].split_at_mut(width);
    let upper = upper.as_chunks_mut::<16>().0;
    let lower = lower.as_chunks::<16>().0;
    for x in 0..upper.len().min(lower.len()) {
        let result = reduce(&upper[x], &lower[x]);
        let (output, _) = upper[x / 2].as_chunks_mut::<8>();
        output[x % 2] = result;
    }

    for y in 1..height / 2 {
        let (output, source) = samples.split_at_mut(2 * y * width);
        let (upper, lower) = source[..2 * width].split_at(width);
        let start = y * half_width;
        let row = &mut output[start..start + half_width];
        for ((upper, lower), output) in upper
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lower.as_chunks::<16>().0)
            .zip(row.as_chunks_mut::<8>().0)
        {
            *output = reduce(upper, lower);
        }
    }
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn halve_in_place(samples: &mut [f32], width: usize, height: usize) {
    let half_width = width / 2;

    // The first output row overlaps the upper input row. Each output pixel
    // replaces a sample that has already been read.
    let (upper, rest) = samples.split_at_mut(width);
    for (x, lower) in rest[..width].as_chunks::<2>().0.iter().enumerate() {
        let a = [upper[2 * x], upper[2 * x + 1]];
        upper[x] = average_float(&a, lower);
    }

    // Later output rows are entirely before their input rows.
    for y in 1..height / 2 {
        let (output, source) = samples.split_at_mut(2 * y * width);
        let (upper, lower) = source[..2 * width].split_at(width);
        let start = y * half_width;
        let row = &mut output[start..start + half_width];
        for (dst, value) in row.iter_mut().zip(reduce_row(upper, lower, average_float)) {
            *dst = value;
        }
    }
}

fn reduced_rows<'a, T>(
    input: Plane<'a, T>,
    average: impl Fn(&[T; 2], &[T; 2]) -> f32 + Copy + 'a,
) -> impl Iterator<Item = impl Iterator<Item = f32>> {
    let width = input.width;
    (0..input.height / 2).map(move |y| {
        let start = 2 * y * input.stride;
        let upper = &input.samples[start..start + width];
        let lower = &input.samples[start + input.stride..start + input.stride + width];
        reduce_row(upper, lower, average)
    })
}

fn reduce_row<'a, T, U>(
    upper: &'a [T],
    lower: &'a [T],
    average: impl Fn(&[T; 2], &[T; 2]) -> U + 'a,
) -> impl Iterator<Item = U> + 'a {
    upper
        .as_chunks::<2>()
        .0
        .iter()
        .zip(lower.as_chunks::<2>().0)
        .map(move |(a, b)| average(a, b))
}

#[inline]
fn average_float(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    0.25 * (a[0] + a[1] + b[0] + b[1])
}

#[inline]
fn average_byte(a: &[u8; 2], b: &[u8; 2]) -> f32 {
    normalize_sum(byte_sum(a, b))
}

#[inline]
fn byte_sum(a: &[u8; 2], b: &[u8; 2]) -> u16 {
    u16::from(a[0]) + u16::from(a[1]) + u16::from(b[0]) + u16::from(b[1])
}

#[inline]
pub(crate) fn normalize_sum(sum: u16) -> f32 {
    f32::from(sum) * (1.0 / 1020.0)
}

#[inline]
fn average_byte_sums(a: &[u16; 2], b: &[u16; 2]) -> f32 {
    average_float(&a.map(normalize_sum), &b.map(normalize_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_tiles_allocate_only_the_first_reduction() {
        let region = Region {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        };
        let area = region.width * region.height;
        let bytes = [127; 32 * 32];
        let floats = [0.5; 32 * 32];
        let kernels = Kernels::new();
        let inputs = [
            GrayImage::u8(&bytes, 32, 32, 32).unwrap(),
            GrayImage::unit_f32(&floats, 32, 32, 32).unwrap(),
        ];

        for input in inputs {
            let mut scratch = Workspace::default();
            let _ = TilePyramid::new(input, region, &mut scratch).unwrap();
            assert_eq!(scratch.allocated_bytes(), 0);

            let mut tile = TilePyramid::new(input, region, &mut scratch).unwrap();
            tile.halve(kernels);
            tile.halve(kernels);
            assert_eq!(scratch.floats.len(), area / 4);
            assert_eq!(scratch.bytes.capacity(), 0);
        }
    }

    #[test]
    fn in_place_float_reduction_matches_the_separate_buffer() {
        let values = [0.0, -0.0, 1.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE];
        for (width, height) in [(16, 2), (32, 16), (48, 6)] {
            let region = Region {
                x: 0,
                y: 0,
                width,
                height,
            };
            let mut samples: Vec<_> = (0..width * height)
                .map(|i| values[(i + i / 13) % values.len()])
                .collect();
            let mut expected = vec![0.0; samples.len() / 4];
            halve(
                Plane::packed(&samples, region),
                &mut expected,
                average_float,
            );
            Kernels::new().halve_float_in_place(&mut samples, width, height);
            for (actual, expected) in samples.iter().zip(expected) {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn strided_byte_reduction_matches_scalar() {
        let (width, height, stride) = (48, 6, 53);
        let samples: Vec<_> = (0..(height - 1) * stride + width)
            .map(|i| (i * 37) as u8)
            .collect();
        let input = Plane {
            samples: &samples,
            width,
            height,
            stride,
        };
        let mut expected = vec![0.0; width * height / 4];
        let mut actual = expected.clone();
        halve(input, &mut expected, average_byte);
        Kernels::new().halve_bytes(input, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn prepared_levels_store_only_one_copy_of_replicated_edges() {
        let mut bytes = [0; 6 * 4];
        for (i, sample) in bytes.iter_mut().enumerate() {
            *sample = (i * 11) as u8;
        }
        let image = GrayImage::u8(&bytes, 6, 4, 6).unwrap();
        let layout = Layout::new(6, 4, crate::MAX_SCALES).unwrap();
        let prepared = prepare(image.to_packed().unwrap(), layout, layout.scales, false);
        let Level::Byte(finest) = &prepared.finest else {
            panic!("expected a byte level");
        };
        assert_eq!((finest.width, finest.height()), (6, 4));
        let ReducedLevel::ByteSums(first) = &prepared.reduced[0] else {
            panic!("expected a level of byte sums");
        };
        assert_eq!((first.width, first.height()), (4, 3));

        let mut prepared_scratch = Workspace::default();
        let mut streamed_scratch = Workspace::default();
        let mut region = layout.tiles().next().unwrap();
        let mut streamed = TilePyramid::new(image, region, &mut streamed_scratch).unwrap();
        for scale in 1..layout.scales {
            streamed.halve(Kernels::new());
            region = region.halve();
            let (TilePlane::Float(prepared), TilePlane::Float(streamed)) = (
                prepared.tile(scale, region, &mut prepared_scratch),
                streamed.plane(),
            ) else {
                panic!("expected float tiles");
            };
            assert_eq!(prepared.samples, streamed.samples, "scale {scale}");
        }
    }
}
