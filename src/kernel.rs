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

//! Portable SSIM window kernel and shared deficit calculation.

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
use crate::image::normalize;
use crate::pyramid::WINDOW;
#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
use crate::pyramid::normalize_sum;

pub(crate) type DeficitFn<A = f32, B = A> =
    unsafe fn(&[A], usize, &[B], usize, usize, usize) -> f64;

const C1: f32 = 1e-4;
const C2: f32 = 9e-4;

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn scalar(
    reference: &[f32],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    float_scalar(
        reference,
        reference_stride,
        test,
        test_stride,
        width,
        height,
        |x| x,
    )
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn mixed_scalar(
    reference: &[u8],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    float_scalar(
        reference,
        reference_stride,
        test,
        test_stride,
        width,
        height,
        normalize,
    )
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn byte_sum_scalar(
    reference: &[u16],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    float_scalar(
        reference,
        reference_stride,
        test,
        test_stride,
        width,
        height,
        normalize_sum,
    )
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
fn float_scalar<T: Copy>(
    reference: &[T],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
    convert: impl Fn(T) -> f32,
) -> f64 {
    let mut deficit = 0.0;
    for y in (0..height).step_by(WINDOW) {
        let a = &reference[y * reference_stride..(y + WINDOW - 1) * reference_stride + width];
        let b = &test[y * test_stride..(y + WINDOW - 1) * test_stride + width];
        for x in (0..width).step_by(WINDOW) {
            let (base_a, base_b) = (convert(a[x]), b[x]);
            let base_d = base_a - base_b;
            // Accumulate by column to give the compiler independent SIMD lanes.
            let mut sum_a = [0.0; WINDOW];
            let mut sum_b = [0.0; WINDOW];
            let mut sum_d = [0.0; WINDOW];
            let mut square_sum = [0.0; WINDOW];
            let mut square_d = [0.0; WINDOW];
            for row in 0..WINDOW {
                let start_a = row * reference_stride + x;
                let start_b = row * test_stride + x;
                let a = &a[start_a..start_a + WINDOW];
                let b = &b[start_b..start_b + WINDOW];
                for i in 0..WINDOW {
                    let a = convert(a[i]);
                    let da = a - base_a;
                    let db = b[i] - base_b;
                    let d = (a - b[i]) - base_d;
                    sum_a[i] += da;
                    sum_b[i] += db;
                    sum_d[i] += d;
                    square_sum[i] += da * da + db * db;
                    square_d[i] += d * d;
                }
            }
            deficit += f64::from(window(
                base_a,
                base_b,
                sum_a.into_iter().sum(),
                sum_b.into_iter().sum(),
                sum_d.into_iter().sum(),
                square_sum.into_iter().sum(),
                square_d.into_iter().sum(),
            ));
        }
    }
    deficit
}

#[inline(always)]
pub(crate) fn window(
    base_a: f32,
    base_b: f32,
    sum_a: f32,
    sum_b: f32,
    sum_d: f32,
    square_sum: f32,
    square_d: f32,
) -> f32 {
    const INV_AREA: f32 = 1.0 / (WINDOW * WINDOW) as f32;

    // Center the moments on the first sample to preserve precision in flat windows.
    let (a, b, d) = (sum_a * INV_AREA, sum_b * INV_AREA, sum_d * INV_AREA);
    let var_sum = square_sum * INV_AREA - (a * a + b * b);
    let var_d = square_d * INV_AREA - d * d;
    let (mean_a, mean_b, mean_d) = (base_a + a, base_b + b, (base_a - base_b) + d);
    // Compute the losses directly. Subtracting SSIM from one loses precision
    // on near-identical windows. The conditional only clamps roundoff.
    let var_sum = if var_sum < 0.0 { 0.0 } else { var_sum };
    let var_d = if var_d < 0.0 { 0.0 } else { var_d };
    let luminance = mean_d * mean_d / (mean_a * mean_a + mean_b * mean_b + C1);
    let structure = var_d / (var_sum + C2);
    luminance + (1.0 - luminance) * structure
}

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
pub(crate) fn byte_scalar(
    reference: &[u8],
    reference_stride: usize,
    test: &[u8],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let mut deficit = 0.0;
    for y in (0..height).step_by(WINDOW) {
        let a = &reference[y * reference_stride..(y + WINDOW - 1) * reference_stride + width];
        let b = &test[y * test_stride..(y + WINDOW - 1) * test_stride + width];
        for x in (0..width).step_by(WINDOW) {
            let (mut sum_a, mut sum_b, mut square_sum, mut square_d) = (0, 0, 0, 0);
            for row in 0..WINDOW {
                let start_a = row * reference_stride + x;
                let start_b = row * test_stride + x;
                let a = &a[start_a..start_a + WINDOW];
                let b = &b[start_b..start_b + WINDOW];
                for (&a, &b) in a.iter().zip(b) {
                    let (a, b) = (i32::from(a), i32::from(b));
                    let d = a - b;
                    sum_a += a;
                    sum_b += b;
                    square_sum += a * a + b * b;
                    square_d += d * d;
                }
            }
            deficit += f64::from(byte_window(sum_a, sum_b, square_sum, square_d));
        }
    }
    deficit
}

#[inline(always)]
pub(crate) fn byte_window(sum_a: i32, sum_b: i32, square_sum: i32, square_d: i32) -> f32 {
    if square_d == 0 {
        return 0.0;
    }
    const AREA: i32 = (WINDOW * WINDOW) as i32;
    const SCALE_SQUARED: f32 = (AREA * 255 * AREA * 255) as f32;
    // Keep the variance subtraction exact in byte units. The largest term
    // is 2 × 64² × 255², which fits in i32.
    let mean_squares = sum_a * sum_a + sum_b * sum_b;
    let d = sum_a - sum_b;
    let var_sum = AREA * square_sum - mean_squares;
    let var_d = AREA * square_d - d * d;
    let luminance = (d * d) as f32 / (mean_squares as f32 + C1 * SCALE_SQUARED);
    let structure = var_d as f32 / (var_sum as f32 + C2 * SCALE_SQUARED);
    luminance + (1.0 - luminance) * structure
}
