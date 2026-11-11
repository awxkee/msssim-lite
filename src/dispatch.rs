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

//! CPU dispatch shared by prepared and one-shot scoring.

#[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
use crate::kernel;
use crate::kernel::DeficitFn;
use crate::pyramid::{self, Plane, TilePlane, WINDOW};

type HalveBytesFn = unsafe fn(Plane<'_, u8>, &mut [f32]);

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    deficit: DeficitFn,
    byte_deficit: DeficitFn<u8>,
    mixed_deficit: DeficitFn<u8, f32>,
    halve_bytes: HalveBytesFn,
}

impl Kernels {
    pub(crate) fn new() -> Self {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Self {
                deficit: crate::avx::deficit,
                byte_deficit: crate::avx::byte_deficit,
                mixed_deficit: crate::avx::mixed_deficit,
                halve_bytes: crate::x86::halve_bytes,
            };
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("sse2") {
            return Self {
                deficit: crate::sse::deficit,
                byte_deficit: crate::sse::byte_deficit,
                mixed_deficit: crate::sse::mixed_deficit,
                halve_bytes: crate::x86::halve_bytes,
            };
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            Self {
                deficit: crate::neon::deficit,
                byte_deficit: crate::neon::byte_deficit,
                mixed_deficit: crate::neon::mixed_deficit,
                halve_bytes: pyramid::halve_bytes,
            }
        }
        #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
        {
            Self {
                deficit: kernel::scalar,
                byte_deficit: kernel::byte_scalar,
                mixed_deficit: kernel::mixed_scalar,
                halve_bytes: pyramid::halve_bytes,
            }
        }
    }

    pub(crate) fn deficit(self, reference: TilePlane<'_>, test: TilePlane<'_>) -> f64 {
        match (reference, test) {
            (TilePlane::Byte(a), TilePlane::Byte(b)) => score(self.byte_deficit, a, b),
            (TilePlane::Float(a), TilePlane::Float(b)) => score(self.deficit, a, b),
            (TilePlane::Byte(a), TilePlane::Float(b))
            | (TilePlane::Float(b), TilePlane::Byte(a)) => score(self.mixed_deficit, a, b),
        }
    }

    pub(crate) fn halve_bytes(self, input: Plane<'_, u8>, output: &mut [f32]) {
        assert!(input.width > 0 && input.width.is_multiple_of(2 * WINDOW));
        assert!(input.height > 0 && input.height.is_multiple_of(2));
        check_plane(input.stride, input.width, input.height, input.samples.len());
        assert_eq!(output.len(), (input.width / 2) * (input.height / 2));
        // SAFETY: Dispatch checked CPU features and the slices cover the reduction.
        unsafe { (self.halve_bytes)(input, output) }
    }
}

fn score<A, B>(deficit: DeficitFn<A, B>, reference: Plane<'_, A>, test: Plane<'_, B>) -> f64 {
    let (width, height) = (test.width, test.height);
    assert!(
        width > 0 && height > 0 && width.is_multiple_of(WINDOW) && height.is_multiple_of(WINDOW)
    );
    assert_eq!((reference.width, reference.height), (width, height));
    check_plane(reference.stride, width, height, reference.samples.len());
    check_plane(test.stride, width, height, test.samples.len());
    // SAFETY: Dispatch selects a supported backend. Both planes cover whole windows.
    unsafe {
        deficit(
            reference.samples,
            reference.stride,
            test.samples,
            test.stride,
            width,
            height,
        )
    }
}

fn check_plane(stride: usize, width: usize, height: usize, len: usize) {
    assert!(stride >= width);
    assert!(
        (height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(width))
            .is_some_and(|needed| needed <= len)
    );
}
