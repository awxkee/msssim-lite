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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
fn sum_lanes(value: __m128i) -> i32 {
    let pairs = _mm_add_epi32(value, _mm_shuffle_epi32::<0x4e>(value));
    _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_shuffle_epi32::<0x55>(pairs)))
}

/// # Safety
/// SSE2 must be available. Width and height must be positive multiples
/// of eight. Each stride must be at least the width, and each slice must
/// cover the active rectangle for its stride.
#[target_feature(enable = "sse2")]
pub(crate) fn byte_deficit(
    reference: &[u8],
    reference_stride: usize,
    test: &[u8],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let mut total = 0.0;
    let zero = _mm_setzero_si128();
    // Each window contains eight full rows of eight samples in both images.
    unsafe {
        for y in (0..height).step_by(8) {
            let reference_row = reference.as_ptr().add(y * reference_stride);
            let test_row = test.as_ptr().add(y * test_stride);
            for x in (0..width).step_by(8) {
                let reference_block = reference_row.add(x);
                let test_block = test_row.add(x);
                let (mut sum_a, mut sum_b) = (zero, zero);
                let (mut square_a, mut square_b, mut square_d) = (zero, zero, zero);
                for row in 0..8 {
                    let a = _mm_loadu_si64(reference_block.add(row * reference_stride));
                    let b = _mm_loadu_si64(test_block.add(row * test_stride));
                    sum_a = _mm_add_epi32(sum_a, _mm_sad_epu8(a, zero));
                    sum_b = _mm_add_epi32(sum_b, _mm_sad_epu8(b, zero));
                    let a = _mm_unpacklo_epi8(a, zero);
                    let b = _mm_unpacklo_epi8(b, zero);
                    let d = _mm_sub_epi16(a, b);
                    square_a = _mm_add_epi32(square_a, _mm_madd_epi16(a, a));
                    square_b = _mm_add_epi32(square_b, _mm_madd_epi16(b, b));
                    square_d = _mm_add_epi32(square_d, _mm_madd_epi16(d, d));
                }
                total += f64::from(byte_window(
                    _mm_cvtsi128_si32(sum_a),
                    _mm_cvtsi128_si32(sum_b),
                    sum_lanes(_mm_add_epi32(square_a, square_b)),
                    sum_lanes(square_d),
                ));
            }
        }
    }
    total
}
