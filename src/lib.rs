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

//! Fast block-window multi-scale SSIM on single-channel images, with a
//! prepared-reference API for encoder search loops.
//!
//! This is a proxy, not the reference MS-SSIM: SSIM (luminance times
//! structure) over non-overlapping 8x8 windows, at dyadic scales made with a
//! 2x2 box filter, the scales weighted equally. Scores are not comparable
//! with other MS-SSIM implementations; they order reconstructions of one
//! image, which is what an encoder decision needs.
//!
//! Scores are deficits: 0 is identical, higher is worse. Samples are nominally
//! in `0.0..=1.0`; the stabilizing constants assume that range.
//!
//! ```
//! use msssim_lite::GrayImage;
//! let (w, h) = (64, 48);
//! let reference: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
//! let mut test = reference.clone();
//! test[100] ^= 0x40;
//!
//! let score = msssim_lite::compare(
//!     GrayImage::u8(&reference, w, h, w)?,
//!     GrayImage::u8(&test, w, h, w)?,
//! )?;
//! assert!(score.deficit() > 0.0);
//! # Ok::<(), msssim_lite::Error>(())
//! ```
//!
//! Reuse a [`PreparedReference`] when one reference is compared with many
//! candidates: its padded pyramid is built once.
//!
//! ```
//! use msssim_lite::{GrayImage, PreparedReference};
//! # let (w, h) = (64, 48);
//! # let reference: Vec<f32> = (0..w * h).map(|i| (i * 7 % 251) as f32 / 250.0).collect();
//! # let candidate = reference.clone();
//! let prepared = PreparedReference::new(GrayImage::unit_f32(&reference, w, h, w)?)?;
//! let score = prepared.compare(GrayImage::unit_f32(&candidate, w, h, w)?)?;
//! assert_eq!(score.deficit(), 0.0);
//! # Ok::<(), msssim_lite::Error>(())
//! ```
//!
//! Use [`PreparedReference::incremental`] to score local candidate edits and
//! [`Checkpoint`] to roll back trial updates.
//!
//! Comparisons stream tiles with bounded scratch space. The `neon` feature
//! selects NEON unconditionally on AArch64. On x86, AVX2+FMA and SSE2 are
//! selected at runtime when enabled. A portable fallback is also available.

#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
mod avx;
mod dispatch;
mod image;
mod incremental;
mod kernel;
mod metric;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;
mod pyramid;
#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
mod sse;
#[cfg(all(
    any(feature = "avx", feature = "sse"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod x86;

pub use image::GrayImage;
pub use incremental::{Checkpoint, IncrementalState, Rect};
pub use metric::{DEFAULT_SCALES, MAX_SCALES, PreparedReference, Score};

/// Multi-scale SSIM deficit of `test` against `reference` at
/// [`DEFAULT_SCALES`] scales (0 is identical).
pub fn compare(reference: GrayImage<'_>, test: GrayImage<'_>) -> Result<Score, Error> {
    metric::compare(reference, test)
}

/// Invalid image layout, size or samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Width and height must be nonzero and fit addressable memory.
    InvalidDimensions,
    /// Invalid stride or buffer length, or a different sample format in an incremental update.
    InvalidLayout,
    /// Reference and test dimensions differ.
    DimensionMismatch,
    /// The number of scales must be in `1..=MAX_SCALES`.
    InvalidScales,
    /// A floating-point sample is NaN or infinite.
    NonFiniteSample,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "image dimensions must be nonzero and fit in memory",
            Self::InvalidLayout => "image stride, buffer length or sample format is invalid",
            Self::DimensionMismatch => "reference and test dimensions differ",
            Self::InvalidScales => "the number of scales is out of range",
            Self::NonFiniteSample => "image contains a non-finite sample",
        })
    }
}

impl std::error::Error for Error {}
