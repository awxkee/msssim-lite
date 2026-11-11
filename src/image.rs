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

use crate::Error;
use crate::pyramid::{Layout, Level, PackedLevel, Plane, Region, TilePlane};

#[derive(Clone, Copy)]
enum Samples<'a> {
    U8(&'a [u8]),
    UnitF32(&'a [f32]),
}

/// A borrowed single-channel image. Strides are in samples.
#[derive(Clone, Copy)]
pub struct GrayImage<'a> {
    samples: Samples<'a>,
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a> GrayImage<'a> {
    /// Bytes, read as `value / 255`. Row padding is allowed.
    pub fn u8(
        samples: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, samples.len())?;
        Ok(Self {
            samples: Samples::U8(samples),
            width,
            height,
            stride,
        })
    }

    /// Floats nominally in `0.0..=1.0`. Row padding is allowed.
    pub fn unit_f32(
        samples: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, samples.len())?;
        Ok(Self {
            samples: Samples::UnitF32(samples),
            width,
            height,
            stride,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn is_byte(self) -> bool {
        matches!(self.samples, Samples::U8(_))
    }

    pub(crate) fn from_packed(level: &'a Level, height: usize) -> Self {
        let (samples, width) = match level {
            Level::Byte(level) => (Samples::U8(&level.samples), level.width),
            Level::Float(level) => (Samples::UnitF32(&level.samples), level.width),
        };
        Self {
            samples,
            width,
            height,
            stride: width,
        }
    }

    /// Read only an already clipped, nonempty region of an update.
    pub(crate) fn checked_region(self, region: Region) -> Result<TilePlane<'a>, Error> {
        match self.samples {
            Samples::U8(samples) => Ok(TilePlane::Byte(
                self.interior(samples, region).expect("clipped region"),
            )),
            Samples::UnitF32(samples) => {
                let plane = self.interior(samples, region).expect("clipped region");
                check_plane_finite(plane)?;
                Ok(TilePlane::Float(plane))
            }
        }
    }

    /// Borrow interior tiles. Byte tiles on an edge use the byte scratch.
    /// A float tile on an edge is written by `write_float_region`.
    pub(crate) fn borrow_region<'tile>(
        self,
        region: Region,
        scratch: &'tile mut [u8],
    ) -> Result<Option<TilePlane<'tile>>, Error>
    where
        'a: 'tile,
    {
        match self.samples {
            Samples::U8(samples) => {
                let plane = if let Some(plane) = self.interior(samples, region) {
                    plane
                } else {
                    let scratch = &mut scratch[..region.width * region.height];
                    self.write_rows(samples, region, scratch, |_| Ok(()))?;
                    Plane::packed(scratch, region)
                };
                Ok(Some(TilePlane::Byte(plane)))
            }
            Samples::UnitF32(samples) => {
                let Some(plane) = self.interior(samples, region) else {
                    return Ok(None);
                };
                check_plane_finite(plane)?;
                Ok(Some(TilePlane::Float(plane)))
            }
        }
    }

    fn interior<T>(self, samples: &[T], region: Region) -> Option<Plane<'_, T>> {
        if region.x + region.width > self.width || region.y + region.height > self.height {
            return None;
        }
        let start = region.y * self.stride + region.x;
        let len = (region.height - 1) * self.stride + region.width;
        Some(Plane {
            samples: &samples[start..start + len],
            stride: self.stride,
            width: region.width,
            height: region.height,
        })
    }

    pub(crate) fn write_float_region(
        self,
        region: Region,
        output: &mut [f32],
    ) -> Result<(), Error> {
        let Samples::UnitF32(samples) = self.samples else {
            unreachable!("byte tiles use byte scratch")
        };
        self.write_rows(samples, region, output, check_finite)
    }

    pub(crate) fn padded(self, region: Region) -> Result<Level, Error> {
        let width = region.width;
        match self.samples {
            Samples::U8(samples) => {
                let samples = self.collect_rows(samples, region, |_| Ok(()))?;
                Ok(Level::Byte(PackedLevel { samples, width }))
            }
            Samples::UnitF32(samples) => {
                let samples = self.collect_rows(samples, region, check_finite)?;
                Ok(Level::Float(PackedLevel { samples, width }))
            }
        }
    }

    fn visible_rows<T>(self, samples: &[T], region: Region) -> impl ExactSizeIterator<Item = &[T]> {
        let x = region.x.min(self.width - 1);
        let y = region.y.min(self.height - 1);
        let width = region.width.min(self.width - x);
        let height = region.height.min(self.height - y);
        let start = y * self.stride + x;
        let len = (height - 1) * self.stride + width;
        samples[start..start + len]
            .chunks(self.stride)
            .map(move |row| &row[..width])
    }

    fn write_rows<T: Copy>(
        self,
        samples: &[T],
        region: Region,
        output: &mut [T],
        validate: impl Fn(&[T]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let rows = self.visible_rows(samples, region);
        let (visible, bottom) = output.split_at_mut(rows.len() * region.width);
        for (src, dst) in rows.zip(visible.chunks_exact_mut(region.width)) {
            let (pixels, right) = dst.split_at_mut(src.len());
            validate(src)?;
            pixels.copy_from_slice(src);
            right.fill(pixels[src.len() - 1]);
        }
        let last = &visible[visible.len() - region.width..];
        for dst in bottom.chunks_exact_mut(region.width) {
            dst.copy_from_slice(last);
        }
        Ok(())
    }

    fn collect_rows<T: Copy>(
        self,
        samples: &[T],
        region: Region,
        validate: impl Fn(&[T]) -> Result<(), Error>,
    ) -> Result<Vec<T>, Error> {
        let mut output = Vec::with_capacity(region.width * region.height);
        let rows = self.visible_rows(samples, region);
        let visible_height = rows.len();
        for (y, row) in rows.enumerate() {
            validate(row)?;
            output.extend_from_slice(row);
            output.resize((y + 1) * region.width, row[row.len() - 1]);
        }
        let last = (visible_height - 1) * region.width..visible_height * region.width;
        for _ in visible_height..region.height {
            output.extend_from_within(last.clone());
        }
        Ok(output)
    }
}

fn check_plane_finite(plane: Plane<'_>) -> Result<(), Error> {
    for row in plane.samples.chunks(plane.stride) {
        check_finite(&row[..plane.width])?;
    }
    Ok(())
}

#[inline]
pub(crate) fn normalize(sample: u8) -> f32 {
    f32::from(sample) * (1.0 / 255.0)
}

#[inline]
fn check_finite(row: &[f32]) -> Result<(), Error> {
    // Reduce the row without short-circuiting so the compiler can vectorize it.
    if row.iter().fold(true, |ok, v| ok & v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFiniteSample)
    }
}

fn validate(width: usize, height: usize, stride: usize, len: usize) -> Result<(), Error> {
    Layout::new(width, height, 1)?;
    if stride < width {
        return Err(Error::InvalidLayout);
    }
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .ok_or(Error::InvalidLayout)?;
    if len < needed {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}
