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

use crate::image::normalize;
use crate::kernel::window;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
fn sum_lanes(value: __m128) -> f32 {
    let pairs = _mm_add_ps(value, _mm_movehl_ps(value, value));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
}

/// # Safety
/// SSE2 must be available. Width and height must be positive multiples
/// of eight. Each stride must be at least the width, and each slice must
/// cover the active rectangle for its stride.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn deficit(
    reference: &[f32],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    // SAFETY: This specialization reads four bytes per reference sample.
    unsafe {
        score::<false>(
            reference.as_ptr().cast(),
            reference_stride,
            test.as_ptr(),
            test_stride,
            width,
            height,
        )
    }
}

/// # Safety
/// SSE2 must be available. Width and height must be positive multiples
/// of eight. Each stride must be at least the width, and each slice must
/// cover the active rectangle for its stride.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn mixed_deficit(
    reference: &[u8],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    // SAFETY: This specialization reads one byte per reference sample.
    unsafe {
        score::<true>(
            reference.as_ptr(),
            reference_stride,
            test.as_ptr(),
            test_stride,
            width,
            height,
        )
    }
}

#[target_feature(enable = "sse2")]
unsafe fn score<const BYTE: bool>(
    reference: *const u8,
    reference_stride: usize,
    test: *const f32,
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let sample_size = if BYTE { 1 } else { size_of::<f32>() };
    let reference_stride = reference_stride * sample_size;
    let mut total = 0.0;
    // Each window contains eight full rows of eight samples in both images.
    unsafe {
        for y in (0..height).step_by(8) {
            let reference_row = reference.add(y * reference_stride);
            let test_row = test.add(y * test_stride);
            for x in (0..width).step_by(8) {
                let reference_block = reference_row.add(x * sample_size);
                let test_block = test_row.add(x);
                let base_a = if BYTE {
                    normalize(*reference_block)
                } else {
                    *reference_block.cast::<f32>()
                };
                let base_b = *test_block;
                let base_av = _mm_set1_ps(base_a);
                let base_bv = _mm_set1_ps(base_b);
                let base_dv = _mm_set1_ps(base_a - base_b);

                let zero = _mm_setzero_ps();
                let (mut sum_a, mut sum_b, mut sum_d) = (zero, zero, zero);
                let (mut square_sum, mut square_d) = (zero, zero);
                for row in 0..8 {
                    let reference_pixels = reference_block.add(row * reference_stride);
                    let test_pixels = test_block.add(row * test_stride);
                    let halves = if BYTE {
                        let zero = _mm_setzero_si128();
                        let words = _mm_unpacklo_epi8(_mm_loadu_si64(reference_pixels), zero);
                        let scale = _mm_set1_ps(normalize(1));
                        [
                            _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(words, zero)), scale),
                            _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(words, zero)), scale),
                        ]
                    } else {
                        let a = reference_pixels.cast::<f32>();
                        [_mm_loadu_ps(a), _mm_loadu_ps(a.add(4))]
                    };
                    let test_halves = [_mm_loadu_ps(test_pixels), _mm_loadu_ps(test_pixels.add(4))];
                    for (a, b) in halves.into_iter().zip(test_halves) {
                        let d = _mm_sub_ps(_mm_sub_ps(a, b), base_dv);
                        let a = _mm_sub_ps(a, base_av);
                        let b = _mm_sub_ps(b, base_bv);
                        sum_a = _mm_add_ps(sum_a, a);
                        sum_b = _mm_add_ps(sum_b, b);
                        sum_d = _mm_add_ps(sum_d, d);
                        let squares = _mm_add_ps(_mm_mul_ps(a, a), _mm_mul_ps(b, b));
                        square_sum = _mm_add_ps(square_sum, squares);
                        square_d = _mm_add_ps(square_d, _mm_mul_ps(d, d));
                    }
                }
                total += f64::from(window(
                    base_a,
                    base_b,
                    sum_lanes(sum_a),
                    sum_lanes(sum_b),
                    sum_lanes(sum_d),
                    sum_lanes(square_sum),
                    sum_lanes(square_d),
                ));
            }
        }
    }
    total
}
