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

use crate::kernel::byte_window;
use std::arch::aarch64::*;

/// # Safety
/// The CPU must support NEON. Width and height must be positive multiples
/// of eight. Each stride must be at least the width, and each slice must
/// cover its active rectangle through the last pixel of the last row.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn byte_deficit(
    reference: &[u8],
    reference_stride: usize,
    test: &[u8],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let mut deficit = 0.0;
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            let (mut sum_a, mut sum_b) = (vdupq_n_u16(0), vdupq_n_u16(0));
            let (mut square_a, mut square_b, mut square_d) =
                (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
            // SAFETY: the complete eight-byte span on each row belongs to
            // the active rectangle in each input.
            unsafe {
                let a = reference.as_ptr().add(y * reference_stride + x);
                let b = test.as_ptr().add(y * test_stride + x);
                for row in 0..8 {
                    let a = vld1_u8(a.add(row * reference_stride));
                    let b = vld1_u8(b.add(row * test_stride));
                    let d = vabd_u8(a, b);
                    sum_a = vaddw_u8(sum_a, a);
                    sum_b = vaddw_u8(sum_b, b);
                    square_a = vpadalq_u16(square_a, vmull_u8(a, a));
                    square_b = vpadalq_u16(square_b, vmull_u8(b, b));
                    square_d = vpadalq_u16(square_d, vmull_u8(d, d));
                }
            }
            let square_d = vaddvq_u32(square_d);
            if square_d != 0 {
                // The largest sample sum is 64 × 255, which fits in u16.
                deficit += f64::from(byte_window(
                    i32::from(vaddvq_u16(sum_a)),
                    i32::from(vaddvq_u16(sum_b)),
                    vaddvq_u32(vaddq_u32(square_a, square_b)) as i32,
                    square_d as i32,
                ));
            }
        }
    }
    deficit
}

/// # Safety
/// The CPU must support NEON and the dot-product extension. Width and height
/// must be positive multiples of eight. Each stride must be at least the
/// width, and each slice must cover the active rectangle for its stride.
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon,dotprod")]
pub(crate) fn byte_deficit_dot(
    reference: &[u8],
    reference_stride: usize,
    test: &[u8],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    if width == 8 {
        return byte_deficit(
            reference,
            reference_stride,
            test,
            test_stride,
            width,
            height,
        );
    }

    let mut deficit = 0.0;
    let paired_width = width & !15;
    for y in (0..height).step_by(8) {
        let reference_row = &reference[y * reference_stride..];
        let test_row = &test[y * test_stride..];
        for x in (0..paired_width).step_by(16) {
            let (mut sum_a, mut sum_b) = (vdupq_n_u16(0), vdupq_n_u16(0));
            let (mut square_sum, mut square_d) = (vdupq_n_u32(0), vdupq_n_u32(0));
            // SAFETY: Each load covers 16 active samples belonging to two
            // adjacent windows in the same row.
            unsafe {
                let a = reference_row.as_ptr().add(x);
                let b = test_row.as_ptr().add(x);
                for row in 0..8 {
                    let a = vld1q_u8(a.add(row * reference_stride));
                    let b = vld1q_u8(b.add(row * test_stride));
                    let d = vabdq_u8(a, b);
                    sum_a = vpadalq_u8(sum_a, a);
                    sum_b = vpadalq_u8(sum_b, b);
                    square_sum = dot_squares(square_sum, a);
                    square_sum = dot_squares(square_sum, b);
                    square_d = dot_squares(square_d, d);
                }
            }
            let d = vpadd_u32(vget_low_u32(square_d), vget_high_u32(square_d));
            let first = vget_lane_u32::<0>(d);
            if first != 0 {
                deficit += f64::from(byte_window(
                    i32::from(vaddv_u16(vget_low_u16(sum_a))),
                    i32::from(vaddv_u16(vget_low_u16(sum_b))),
                    vaddv_u32(vget_low_u32(square_sum)) as i32,
                    first as i32,
                ));
            }
            let second = vget_lane_u32::<1>(d);
            if second != 0 {
                deficit += f64::from(byte_window(
                    i32::from(vaddv_u16(vget_high_u16(sum_a))),
                    i32::from(vaddv_u16(vget_high_u16(sum_b))),
                    vaddv_u32(vget_high_u32(square_sum)) as i32,
                    second as i32,
                ));
            }
        }
        if paired_width < width {
            deficit += byte_deficit(
                &reference_row[paired_width..],
                reference_stride,
                &test_row[paired_width..],
                test_stride,
                8,
                8,
            );
        }
    }
    deficit
}

#[cfg(target_endian = "little")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot_squares(mut accumulator: uint32x4_t, pixels: uint8x16_t) -> uint32x4_t {
    // SAFETY: Dot product support was checked by dispatch. Use inline asm to
    // support Rust 1.94, before the corresponding intrinsic was stabilized.
    unsafe {
        std::arch::asm!(
            "udot {accumulator:v}.4s, {pixels:v}.16b, {pixels:v}.16b",
            accumulator = inout(vreg) accumulator,
            pixels = in(vreg) pixels,
            options(nomem, nostack, pure, preserves_flags),
        );
    }
    accumulator
}

#[cfg(all(test, target_endian = "little"))]
mod tests {
    use super::*;

    #[test]
    fn dot_product_scores_match_baseline_neon() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        for width in [8, 16, 24, 32] {
            let height = 16;
            let a_stride = width + 3;
            let b_stride = width + 7;
            let a: Vec<_> = (0..(height - 1) * a_stride + width)
                .map(|i| (i * 117) as u8)
                .collect();
            let mut b = vec![255; (height - 1) * b_stride + width];
            for row in 0..height {
                b[row * b_stride..][..width].copy_from_slice(&a[row * a_stride..][..width]);
            }
            // Exercise both lanes of a window pair and the unpaired tail.
            for column in [0, width / 2, width - 1] {
                b[8 * b_stride + column] ^= 255;
            }

            // SAFETY: Dot product support was checked, and the strided
            // rectangles cover each window including the last row.
            let (baseline, dot_product) = unsafe {
                (
                    byte_deficit(&a, a_stride, &b, b_stride, width, height),
                    byte_deficit_dot(&a, a_stride, &b, b_stride, width, height),
                )
            };
            assert_eq!(baseline.to_bits(), dot_product.to_bits(), "width {width}");
        }
    }
}
