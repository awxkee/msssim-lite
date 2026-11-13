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

use crate::pyramid::{Plane, halve_float_blocks};
use std::arch::aarch64::*;

/// # Safety
/// NEON must be available. Input must cover its active rectangle, with a
/// positive width divisible by 16 and a positive even height. Output must
/// hold exactly one sample per 2×2 input block.
#[target_feature(enable = "neon")]
pub(crate) fn halve_bytes(input: Plane<'_, u8>, output: &mut [f32]) {
    let width = input.width;
    let stride = input.stride;
    let scale = vdupq_n_f32(1.0 / 1020.0);
    for (y, dst) in output.chunks_exact_mut(width / 2).enumerate() {
        let start = 2 * y * stride;
        let upper = &input.samples[start..start + width];
        let lower = &input.samples[start + stride..start + stride + width];
        for ((upper, lower), dst) in upper
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lower.as_chunks::<16>().0)
            .zip(dst.as_chunks_mut::<8>().0)
        {
            // SAFETY: Each source array contains 16 bytes, and both stores
            // together write the eight floats of the destination array.
            unsafe {
                let a = vld1q_u8(upper.as_ptr());
                let b = vld1q_u8(lower.as_ptr());
                let sum = vpadalq_u8(vpaddlq_u8(a), b);
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(sum)));
                let hi = vcvtq_f32_u32(vmovl_high_u16(sum));
                vst1q_f32(dst.as_mut_ptr(), vmulq_f32(lo, scale));
                vst1q_f32(dst.as_mut_ptr().add(4), vmulq_f32(hi, scale));
            }
        }
    }
}

/// # Safety
/// NEON must be available. Width must be a positive multiple of 16 and
/// height must be positive and even. The slice must hold the packed image.
#[target_feature(enable = "neon")]
pub(crate) fn halve_float_in_place(samples: &mut [f32], width: usize, height: usize) {
    halve_float_blocks(samples, width, height, |upper, lower| {
        let mut output = [0.0; 8];
        // SAFETY: Each input has 16 samples and the two stores fill the output.
        unsafe {
            let first = reduce_pairs(vld2q_f32(upper.as_ptr()), vld2q_f32(lower.as_ptr()));
            let second = reduce_pairs(
                vld2q_f32(upper.as_ptr().add(8)),
                vld2q_f32(lower.as_ptr().add(8)),
            );
            vst1q_f32(output.as_mut_ptr(), first);
            vst1q_f32(output.as_mut_ptr().add(4), second);
        }
        output
    });
}

#[inline]
#[target_feature(enable = "neon")]
fn reduce_pairs(upper: float32x4x2_t, lower: float32x4x2_t) -> float32x4_t {
    // Keep the same addition order as the scalar box filter.
    let sum = vaddq_f32(vaddq_f32(vaddq_f32(upper.0, upper.1), lower.0), lower.1);
    vmulq_n_f32(sum, 0.25)
}
