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

    fn tile_area(self) -> usize {
        self.width.min(TILE_WIDTH) * self.height.min(self.tile_height())
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
}

pub(crate) struct PackedLevel<T> {
    pub(crate) samples: Vec<T>,
    pub(crate) width: usize,
}

impl<T> PackedLevel<T> {
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

pub(crate) fn prepare(
    image: GrayImage<'_>,
    layout: Layout,
    kernels: Kernels,
) -> Result<Vec<Level>, Error> {
    let mut levels = Vec::with_capacity(layout.scales);
    let mut region = Region {
        x: 0,
        y: 0,
        width: layout.width,
        height: layout.height,
    };
    levels.push(image.padded(region)?);
    for _ in 1..layout.scales {
        let input = levels[levels.len() - 1].region(region);
        region = region.halve();
        let area = region.width * region.height;
        let samples = match input {
            TilePlane::Byte(input) => {
                let mut samples = vec![0.0; area];
                kernels.halve_bytes(input, &mut samples);
                samples
            }
            TilePlane::Float(input) => {
                let mut samples = Vec::with_capacity(area);
                for row in reduced_rows(input, average_float) {
                    samples.extend(row);
                }
                samples
            }
        };
        levels.push(Level::Float(PackedLevel {
            samples,
            width: region.width,
        }));
    }
    Ok(levels)
}

#[derive(Default)]
pub(crate) struct Workspace {
    floats: Vec<f32>,
    bytes: Vec<u8>,
}

impl Workspace {
    pub(crate) fn new(layout: Layout, byte_input: bool) -> Self {
        Self::with_area(layout.tile_area(), layout.scales, byte_input)
    }

    pub(crate) fn with_area(area: usize, scales: usize, byte_input: bool) -> Self {
        let floats = if scales > 1 {
            // A byte input first writes floats at the half-size level.
            if byte_input {
                area / 4 + area / 16
            } else {
                area + area / 4
            }
        } else if byte_input {
            0
        } else {
            area
        };
        Self {
            floats: vec![0.0; floats],
            bytes: if byte_input {
                vec![0; area]
            } else {
                Vec::new()
            },
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.floats.capacity() * size_of::<f32>() + self.bytes.capacity()
    }
}

/// A tile's current level. The two buffers alternate as the tile shrinks.
pub(crate) struct TilePyramid<'a> {
    borrowed: Option<TilePlane<'a>>,
    buffers: [&'a mut [f32]; 2],
    region: Region,
}

impl<'a> TilePyramid<'a> {
    pub(crate) fn new(
        image: GrayImage<'a>,
        region: Region,
        scratch: &'a mut Workspace,
    ) -> Result<Self, Error> {
        let area = region.width * region.height;
        let borrowed = image.borrow_region(region, &mut scratch.bytes)?;
        let (first, second) = if matches!(borrowed, Some(TilePlane::Byte(_))) {
            // The first reduction writes to the second buffer.
            let split = (area / 16).min(scratch.floats.len());
            scratch.floats.split_at_mut(split)
        } else {
            scratch.floats.split_at_mut(area)
        };
        if borrowed.is_none() {
            image.write_float_region(region, first)?;
        }
        Ok(Self {
            borrowed,
            buffers: [first, second],
            region,
        })
    }

    pub(crate) fn with_patch(
        reference: TilePlane<'_>,
        patch: TilePlane<'_>,
        region: Region,
        footprint: Region,
        scratch: &'a mut Workspace,
    ) -> Self {
        let area = region.width * region.height;
        let split = if matches!(reference, TilePlane::Byte(_)) {
            (area / 16).min(scratch.floats.len())
        } else {
            area
        };
        let (first, second) = scratch.floats.split_at_mut(split);
        let borrowed = match (reference, patch) {
            (TilePlane::Byte(reference), TilePlane::Byte(patch)) => {
                let tile = &mut scratch.bytes[..area];
                overlay_patch(reference, patch, region, footprint, tile);
                Some(TilePlane::Byte(Plane::packed(tile, region)))
            }
            (TilePlane::Float(reference), TilePlane::Float(patch)) => {
                overlay_patch(reference, patch, region, footprint, first);
                None
            }
            _ => unreachable!("patch format was checked"),
        };
        Self {
            borrowed,
            buffers: [first, second],
            region,
        }
    }

    pub(crate) fn plane(&self) -> TilePlane<'_> {
        self.borrowed
            .unwrap_or_else(|| TilePlane::Float(Plane::packed(self.buffers[0], self.region)))
    }

    pub(crate) fn halve(&mut self, kernels: Kernels) {
        let [first, second] = &mut self.buffers;
        let fine = self.region;
        self.region = fine.halve();
        let output = &mut second[..self.region.width * self.region.height];
        match self.borrowed.take() {
            Some(TilePlane::Byte(input)) => kernels.halve_bytes(input, output),
            Some(TilePlane::Float(input)) => halve(input, output, average_float),
            None => halve(Plane::packed(first, fine), output, average_float),
        }
        self.buffers.swap(0, 1);
    }
}

fn overlay_patch<T: Copy>(
    reference: Plane<'_, T>,
    patch: Plane<'_, T>,
    tile: Region,
    footprint: Region,
    output: &mut [T],
) {
    let left = tile.x.max(footprint.x);
    let right = (tile.x + tile.width).min(footprint.x + footprint.width);
    let patch_x = left - footprint.x;
    // The final aligned tile contains both the edge pixel and its replicas.
    let visible = (right - left).min(patch.width - patch_x);
    for (row, (src, dst)) in reference
        .samples
        .chunks(reference.stride)
        .zip(output.chunks_exact_mut(tile.width))
        .enumerate()
    {
        dst.copy_from_slice(&src[..tile.width]);
        let y = tile.y + row;
        if (footprint.y..footprint.y + footprint.height).contains(&y) {
            let patch_row = (y - footprint.y).min(patch.height - 1) * patch.stride;
            let (pixels, padding) = dst[left - tile.x..right - tile.x].split_at_mut(visible);
            pixels.copy_from_slice(&patch.samples[patch_row + patch_x..][..visible]);
            padding.fill(patch.samples[patch_row + patch.width - 1]);
        }
    }
}

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

fn reduced_rows<T>(
    input: Plane<T>,
    average: impl Fn(&[T; 2], &[T; 2]) -> f32 + Copy,
) -> impl Iterator<Item = impl Iterator<Item = f32>> {
    let width = input.width;
    (0..input.height / 2).map(move |y| {
        let start = 2 * y * input.stride;
        let upper = &input.samples[start..start + width];
        let lower = &input.samples[start + input.stride..start + input.stride + width];
        upper
            .as_chunks::<2>()
            .0
            .iter()
            .zip(lower.as_chunks::<2>().0)
            .map(move |(a, b)| average(a, b))
    })
}

#[inline]
fn average_float(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    0.25 * (a[0] + a[1] + b[0] + b[1])
}

#[inline]
fn average_byte(a: &[u8; 2], b: &[u8; 2]) -> f32 {
    let sum = u16::from(a[0]) + u16::from(a[1]) + u16::from(b[0]) + u16::from(b[1]);
    f32::from(sum) * (1.0 / 1020.0)
}
