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
use crate::pyramid::normalize_sum;
use std::arch::aarch64::*;

trait Samples {
    /// Load eight initialized samples with NEON available.
    unsafe fn load_row(ptr: *const Self) -> [float32x4_t; 2];
}

impl Samples for f32 {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_row(ptr: *const Self) -> [float32x4_t; 2] {
        // SAFETY: The caller provides eight readable samples.
        unsafe { [vld1q_f32(ptr), vld1q_f32(ptr.add(4))] }
    }
}

impl Samples for u8 {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_row(ptr: *const Self) -> [float32x4_t; 2] {
        // SAFETY: The caller provides eight readable samples.
        let wide = vmovl_u8(unsafe { vld1_u8(ptr) });
        normalize_words(wide, normalize(1))
    }
}

impl Samples for u16 {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_row(ptr: *const Self) -> [float32x4_t; 2] {
        // SAFETY: The caller provides eight readable samples.
        let wide = unsafe { vld1q_u16(ptr) };
        normalize_words(wide, normalize_sum(1))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn normalize_words(wide: uint16x8_t, scale: f32) -> [float32x4_t; 2] {
    let lower = vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide)));
    let upper = vcvtq_f32_u32(vmovl_high_u16(wide));
    [vmulq_n_f32(lower, scale), vmulq_n_f32(upper, scale)]
}

#[derive(Clone, Copy)]
struct Moments {
    a: float32x4_t,
    b: float32x4_t,
    difference: float32x4_t,
    square_a: float32x4_t,
    square_b: float32x4_t,
    square_difference: float32x4_t,
}

#[inline]
#[target_feature(enable = "neon")]
fn shifted(a: float32x4_t, b: float32x4_t, base: [float32x4_t; 3]) -> [float32x4_t; 3] {
    let difference = vsubq_f32(vsubq_f32(a, b), base[2]);
    [vsubq_f32(a, base[0]), vsubq_f32(b, base[1]), difference]
}

impl Moments {
    #[inline]
    #[target_feature(enable = "neon")]
    fn new(a: float32x4_t, b: float32x4_t, base: [float32x4_t; 3]) -> Self {
        let [a, b, difference] = shifted(a, b, base);
        Self {
            a,
            b,
            difference,
            square_a: vmulq_f32(a, a),
            square_b: vmulq_f32(b, b),
            square_difference: vmulq_f32(difference, difference),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn add(&mut self, a: float32x4_t, b: float32x4_t, base: [float32x4_t; 3]) {
        let [a, b, difference] = shifted(a, b, base);
        self.a = vaddq_f32(self.a, a);
        self.b = vaddq_f32(self.b, b);
        self.difference = vaddq_f32(self.difference, difference);
        self.square_a = vfmaq_f32(self.square_a, a, a);
        self.square_b = vfmaq_f32(self.square_b, b, b);
        self.square_difference = vfmaq_f32(self.square_difference, difference, difference);
    }
}

/// # Safety
/// The CPU must support NEON. Width and height must be nonzero multiples of
/// eight. Each stride must be at least width and each slice must cover its
/// active rectangle through the last pixel of the last row.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn deficit(
    reference: &[f32],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    // SAFETY: The caller supplies the required strided rectangles and NEON.
    unsafe {
        score(
            reference,
            reference_stride,
            test,
            test_stride,
            width,
            height,
        )
    }
}

/// # Safety
/// The CPU must support NEON. Width and height must be nonzero multiples of
/// eight. Each stride must be at least width and each slice must cover its
/// active rectangle through the last pixel of the last row.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mixed_deficit(
    reference: &[u8],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    // SAFETY: The caller supplies the required strided rectangles and NEON.
    unsafe {
        score(
            reference,
            reference_stride,
            test,
            test_stride,
            width,
            height,
        )
    }
}

/// # Safety
/// The CPU must support NEON. Width and height must be nonzero multiples of
/// eight. Each stride must be at least width and each slice must cover its
/// active rectangle through the last pixel of the last row.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn byte_sum_deficit(
    reference: &[u16],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    // SAFETY: The caller supplies the required strided rectangles and NEON.
    unsafe {
        score(
            reference,
            reference_stride,
            test,
            test_stride,
            width,
            height,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn score<S: Samples>(
    reference: &[S],
    reference_stride: usize,
    test: &[f32],
    test_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let mut deficit = 0.0;
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            // SAFETY: The rectangle contains the entire 8×8 window, so both
            // groups of four samples fit on each of the eight rows.
            unsafe {
                let a = reference.as_ptr().add(y * reference_stride + x);
                let b = test.as_ptr().add(y * test_stride + x);
                let [first_a, second_a] = S::load_row(a);
                let [first_b, second_b] = f32::load_row(b);
                let (base_a, base_b) = (vgetq_lane_f32::<0>(first_a), vgetq_lane_f32::<0>(first_b));
                let base = [
                    vdupq_n_f32(base_a),
                    vdupq_n_f32(base_b),
                    vdupq_n_f32(base_a - base_b),
                ];
                // Accumulate the left and right halves independently.
                let mut left = Moments::new(first_a, first_b, base);
                let mut right = Moments::new(second_a, second_b, base);
                for row in 1..8 {
                    let [left_a, right_a] = S::load_row(a.add(row * reference_stride));
                    let [left_b, right_b] = f32::load_row(b.add(row * test_stride));
                    left.add(left_a, left_b, base);
                    right.add(right_a, right_b, base);
                }
                deficit += f64::from(window(
                    base_a,
                    base_b,
                    vaddvq_f32(vaddq_f32(left.a, right.a)),
                    vaddvq_f32(vaddq_f32(left.b, right.b)),
                    vaddvq_f32(vaddq_f32(left.difference, right.difference)),
                    vaddvq_f32(vaddq_f32(
                        vaddq_f32(left.square_a, right.square_a),
                        vaddq_f32(left.square_b, right.square_b),
                    )),
                    vaddvq_f32(vaddq_f32(left.square_difference, right.square_difference)),
                ));
            }
        }
    }
    deficit
}
