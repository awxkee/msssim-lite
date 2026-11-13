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

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn sum_lanes(value: __m128i) -> i32 {
    let pairs = _mm_add_epi32(value, _mm_shuffle_epi32::<0x4e>(value));
    _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_shuffle_epi32::<0x55>(pairs)))
}

/// # Safety
/// AVX2 must be available. Width and height must be positive multiples
/// of eight. Each stride must be at least the width, and each slice must
/// cover the active rectangle for its stride.
#[target_feature(enable = "avx2")]
pub(crate) fn byte_deficit(
    reference: &[u8],
    reference_stride: usize,
    test: &[u8],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let mut total = 0.0;
    let paired_width = width & !15;
    // Each pair and its optional tail are fully inside the active rectangle.
    unsafe {
        for y in (0..height).step_by(8) {
            let reference_row = reference.as_ptr().add(y * reference_stride);
            let test_row = test.as_ptr().add(y * test_stride);
            for x in (0..paired_width).step_by(16) {
                total += score_windows::<true>(
                    reference_row.add(x),
                    reference_stride,
                    test_row.add(x),
                    test_stride,
                );
            }
            if paired_width < width {
                total += score_windows::<false>(
                    reference_row.add(paired_width),
                    reference_stride,
                    test_row.add(paired_width),
                    test_stride,
                );
            }
        }
    }
    total
}

#[inline]
#[target_feature(enable = "avx2")]
fn score_windows<const PAIR: bool>(
    reference: *const u8,
    reference_stride: usize,
    test: *const u8,
    test_stride: usize,
) -> f64 {
    let zero = _mm_setzero_si128();
    let zero_wide = _mm256_setzero_si256();
    let (mut sum_a, mut sum_b) = (zero, zero);
    let (mut square_a, mut square_b, mut square_d) = (zero_wide, zero_wide, zero_wide);
    // The caller provides eight rows of 16 pixels for a pair or eight pixels
    // for a single window. Each 128-bit lane holds one window's widened pixels.
    unsafe {
        for row in 0..8 {
            let a = reference.add(row * reference_stride);
            let b = test.add(row * test_stride);
            let (a, b) = if PAIR {
                (_mm_loadu_si128(a.cast()), _mm_loadu_si128(b.cast()))
            } else {
                (_mm_loadu_si64(a), _mm_loadu_si64(b))
            };
            sum_a = _mm_add_epi32(sum_a, _mm_sad_epu8(a, zero));
            sum_b = _mm_add_epi32(sum_b, _mm_sad_epu8(b, zero));
            let a = _mm256_cvtepu8_epi16(a);
            let b = _mm256_cvtepu8_epi16(b);
            let d = _mm256_sub_epi16(a, b);
            square_a = _mm256_add_epi32(square_a, _mm256_madd_epi16(a, a));
            square_b = _mm256_add_epi32(square_b, _mm256_madd_epi16(b, b));
            square_d = _mm256_add_epi32(square_d, _mm256_madd_epi16(d, d));
        }
    }
    let square_sum = _mm256_add_epi32(square_a, square_b);
    let first = byte_window(
        _mm_cvtsi128_si32(sum_a),
        _mm_cvtsi128_si32(sum_b),
        sum_lanes(_mm256_castsi256_si128(square_sum)),
        sum_lanes(_mm256_castsi256_si128(square_d)),
    );
    if PAIR {
        let second = byte_window(
            _mm_cvtsi128_si32(_mm_srli_si128::<8>(sum_a)),
            _mm_cvtsi128_si32(_mm_srli_si128::<8>(sum_b)),
            sum_lanes(_mm256_extracti128_si256::<1>(square_sum)),
            sum_lanes(_mm256_extracti128_si256::<1>(square_d)),
        );
        f64::from(first) + f64::from(second)
    } else {
        f64::from(first)
    }
}
