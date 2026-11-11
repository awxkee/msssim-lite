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

use crate::pyramid::Plane;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// # Safety
/// SSE2 must be available. Input must cover its active rectangle, with a
/// positive width divisible by 16 and a positive even height. Output must
/// hold exactly one sample per 2×2 input block.
#[target_feature(enable = "sse2")]
pub(crate) fn halve_bytes(input: Plane<'_, u8>, output: &mut [f32]) {
    let width = input.width;
    let stride = input.stride;
    let zero = _mm_setzero_si128();
    let ones = _mm_set1_epi16(1);
    let scale = _mm_set1_ps(1.0 / 1020.0);
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
                let a = _mm_loadu_si128(upper.as_ptr().cast());
                let b = _mm_loadu_si128(lower.as_ptr().cast());
                let lo = _mm_add_epi16(_mm_unpacklo_epi8(a, zero), _mm_unpacklo_epi8(b, zero));
                let hi = _mm_add_epi16(_mm_unpackhi_epi8(a, zero), _mm_unpackhi_epi8(b, zero));
                let lo = _mm_cvtepi32_ps(_mm_madd_epi16(lo, ones));
                let hi = _mm_cvtepi32_ps(_mm_madd_epi16(hi, ones));
                _mm_storeu_ps(dst.as_mut_ptr(), _mm_mul_ps(lo, scale));
                _mm_storeu_ps(dst.as_mut_ptr().add(4), _mm_mul_ps(hi, scale));
            }
        }
    }
}
