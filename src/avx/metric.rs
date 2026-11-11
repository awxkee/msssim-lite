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
#[target_feature(enable = "avx2")]
fn sum_lanes(value: __m256) -> f32 {
    let halves = _mm_add_ps(
        _mm256_castps256_ps128(value),
        _mm256_extractf128_ps::<1>(value),
    );
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
}

/// # Safety
/// AVX2 and FMA must be available. Width and height must be positive
/// multiples of eight. Each stride must be at least the width, and each
/// slice must cover the active rectangle for its stride.
#[target_feature(enable = "avx2,fma")]
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
/// AVX2 and FMA must be available. Width and height must be positive
/// multiples of eight. Each stride must be at least the width, and each
/// slice must cover the active rectangle for its stride.
#[target_feature(enable = "avx2,fma")]
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

#[target_feature(enable = "avx2,fma")]
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
                let base_av = _mm256_set1_ps(base_a);
                let base_bv = _mm256_set1_ps(base_b);
                let base_dv = _mm256_set1_ps(base_a - base_b);

                let zero = _mm256_setzero_ps();
                let (mut sum_a, mut sum_b, mut sum_d) = (zero, zero, zero);
                let (mut square_a, mut square_b, mut square_d) = (zero, zero, zero);
                for row in 0..8 {
                    let a = reference_block.add(row * reference_stride);
                    let a = if BYTE {
                        let bytes = _mm_loadu_si64(a);
                        let words = _mm256_cvtepu8_epi32(bytes);
                        _mm256_mul_ps(_mm256_cvtepi32_ps(words), _mm256_set1_ps(normalize(1)))
                    } else {
                        _mm256_loadu_ps(a.cast())
                    };
                    let b = _mm256_loadu_ps(test_block.add(row * test_stride));
                    let d = _mm256_sub_ps(_mm256_sub_ps(a, b), base_dv);
                    let a = _mm256_sub_ps(a, base_av);
                    let b = _mm256_sub_ps(b, base_bv);
                    sum_a = _mm256_add_ps(sum_a, a);
                    sum_b = _mm256_add_ps(sum_b, b);
                    sum_d = _mm256_add_ps(sum_d, d);
                    square_a = _mm256_fmadd_ps(a, a, square_a);
                    square_b = _mm256_fmadd_ps(b, b, square_b);
                    square_d = _mm256_fmadd_ps(d, d, square_d);
                }
                total += f64::from(window(
                    base_a,
                    base_b,
                    sum_lanes(sum_a),
                    sum_lanes(sum_b),
                    sum_lanes(sum_d),
                    sum_lanes(_mm256_add_ps(square_a, square_b)),
                    sum_lanes(square_d),
                ));
            }
        }
    }
    total
}
