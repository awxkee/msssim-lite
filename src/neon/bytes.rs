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
