//! Behavior of the public API on deterministic images.

use msssim_lite::{DEFAULT_SCALES, Error, GrayImage, MAX_SCALES, PreparedReference};

fn texture(width: usize, height: usize, seed: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            ((x * 3 + y * 5 + (x * y + seed * 31) % 17 * 9) % 256) as u8
        })
        .collect()
}

#[test]
fn identical_images_score_zero_at_any_size() {
    for (w, h) in [(1, 1), (7, 3), (64, 48), (97, 83)] {
        let image = texture(w, h, 1);
        let score = msssim_lite::compare(
            GrayImage::u8(&image, w, h, w).unwrap(),
            GrayImage::u8(&image, w, h, w).unwrap(),
        )
        .unwrap();
        assert_eq!(score.deficit(), 0.0);
        assert_eq!(score.similarity(), 1.0);
        assert_eq!(score.per_scale().len(), 3);
    }
}

#[test]
fn stronger_distortion_scores_worse() {
    let (w, h) = (97, 83);
    let reference = texture(w, h, 1);
    let prepared = PreparedReference::new(GrayImage::u8(&reference, w, h, w).unwrap()).unwrap();
    let posterize = |step: u8| -> Vec<u8> { reference.iter().map(|&v| v / step * step).collect() };
    let mut previous = 0.0;
    for step in [2u8, 8, 32, 64] {
        let test = posterize(step);
        let score = prepared
            .compare(GrayImage::u8(&test, w, h, w).unwrap())
            .unwrap();
        assert!(score.deficit() > previous, "step {step}");
        assert!(score.similarity() < 1.0);
        previous = score.deficit();
    }
}

#[test]
fn u8_and_unit_float_views_agree_and_strides_are_honored() {
    let (w, h, stride) = (50, 37, 64);
    let reference = texture(w, h, 1);
    let test = texture(w, h, 2);
    let pad = |image: &[u8]| -> Vec<f32> {
        let mut out = vec![9.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                out[y * stride + x] = f32::from(image[y * w + x]) / 255.0;
            }
        }
        out
    };
    let bytes = msssim_lite::compare(
        GrayImage::u8(&reference, w, h, w).unwrap(),
        GrayImage::u8(&test, w, h, w).unwrap(),
    )
    .unwrap();
    let floats = msssim_lite::compare(
        GrayImage::unit_f32(&pad(&reference), w, h, stride).unwrap(),
        GrayImage::unit_f32(&pad(&test), w, h, stride).unwrap(),
    )
    .unwrap();
    assert!((bytes.deficit() - floats.deficit()).abs() < 1e-6);
}

#[test]
fn compare_byte_patches_matches_full_scoring_across_tiles_and_edges() {
    let (w, h) = (269, 67);
    let source = texture(w, h, 3);
    let image = GrayImage::u8(&source, w, h, w).unwrap();
    for scales in [1, DEFAULT_SCALES, MAX_SCALES] {
        let reference = PreparedReference::with_scales(image, scales).unwrap();
        for (x, y, pw, ph) in [
            (27, 29, 41, 12),
            (255, 62, 14, 5),
            (w - 1, 0, 1, 1),
            (0, h - 1, 1, 1),
            (w - 1, h - 1, 1, 1),
        ] {
            let stride = pw + 7;
            let mut patch = vec![255; (ph - 1) * stride + pw];
            let mut expected = source.clone();
            for row in 0..ph {
                for col in 0..pw {
                    let pixel = &mut expected[(y + row) * w + x + col];
                    *pixel = pixel.wrapping_add(37);
                    patch[row * stride + col] = *pixel;
                }
            }
            let patch = GrayImage::u8(&patch, pw, ph, stride).unwrap();
            let actual = reference.compare_patch(patch, x, y).unwrap();
            let expected = reference
                .compare(GrayImage::u8(&expected, w, h, w).unwrap())
                .unwrap();
            for (&actual, &expected) in actual.per_scale().iter().zip(expected.per_scale()) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn compare_float_patches_honors_stride_and_validates_the_patch() {
    let (w, h) = (73, 51);
    let source: Vec<_> = texture(w, h, 5)
        .into_iter()
        .map(|v| f32::from(v) / 256.0)
        .collect();
    let reference = PreparedReference::new(GrayImage::unit_f32(&source, w, h, w).unwrap()).unwrap();
    let (x, y, pw, ph, stride) = (27, 29, 46, 22, 59);
    let mut patch = vec![f32::NAN; (ph - 1) * stride + pw];
    let mut expected = source.clone();
    for row in 0..ph {
        for col in 0..pw {
            let sample = &mut expected[(y + row) * w + x + col];
            *sample *= 0.875;
            patch[row * stride + col] = *sample;
        }
    }
    let view = GrayImage::unit_f32(&patch, pw, ph, stride).unwrap();
    let actual = reference.compare_patch(view, x, y).unwrap();
    let expected = reference
        .compare(GrayImage::unit_f32(&expected, w, h, w).unwrap())
        .unwrap();
    for (&actual, &expected) in actual.per_scale().iter().zip(expected.per_scale()) {
        assert!((actual - expected).abs() < 1e-12);
    }

    for (x, y) in [(x + 1, y), (x, y + 1), (usize::MAX, 0), (0, usize::MAX)] {
        assert_eq!(
            reference.compare_patch(view, x, y),
            Err(Error::DimensionMismatch)
        );
    }
    assert_eq!(
        reference.compare_patch(GrayImage::u8(&[0], 1, 1, 1).unwrap(), 0, 0),
        Err(Error::InvalidLayout)
    );
    *patch.last_mut().unwrap() = f32::NAN;
    assert_eq!(
        reference.compare_patch(GrayImage::unit_f32(&patch, pw, ph, stride).unwrap(), x, y),
        Err(Error::NonFiniteSample)
    );
}

#[test]
fn scale_count_is_configurable() {
    let (w, h) = (40, 40);
    let (reference, test) = (texture(w, h, 1), texture(w, h, 2));
    for scales in 1..=MAX_SCALES {
        let prepared =
            PreparedReference::with_scales(GrayImage::u8(&reference, w, h, w).unwrap(), scales)
                .unwrap();
        assert_eq!(prepared.scales(), scales);
        let score = prepared
            .compare(GrayImage::u8(&test, w, h, w).unwrap())
            .unwrap();
        assert_eq!(score.per_scale().len(), scales);
        assert!(score.per_scale().iter().all(|d| d.is_finite() && *d >= 0.0));
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let image = texture(16, 16, 1);
    assert_eq!(
        GrayImage::u8(&image, 0, 16, 16).err(),
        Some(Error::InvalidDimensions)
    );
    assert_eq!(
        GrayImage::u8(&image, 16, 16, 8).err(),
        Some(Error::InvalidLayout)
    );
    assert_eq!(
        GrayImage::u8(&image, 16, 17, 16).err(),
        Some(Error::InvalidLayout)
    );
    let view = GrayImage::u8(&image, 16, 16, 16).unwrap();
    assert_eq!(
        PreparedReference::with_scales(view, 0).err(),
        Some(Error::InvalidScales)
    );
    assert_eq!(
        PreparedReference::with_scales(view, MAX_SCALES + 1).err(),
        Some(Error::InvalidScales)
    );
    let prepared = PreparedReference::new(view).unwrap();
    assert_eq!(
        prepared
            .compare(GrayImage::u8(&image, 8, 16, 16).unwrap())
            .err(),
        Some(Error::DimensionMismatch)
    );
    let nan = vec![f32::NAN; 16 * 16];
    assert_eq!(
        prepared
            .compare(GrayImage::unit_f32(&nan, 16, 16, 16).unwrap())
            .err(),
        Some(Error::NonFiniteSample)
    );
}

fn oracle(
    width: usize,
    height: usize,
    scales: usize,
    reference: impl Fn(usize, usize) -> f64,
    candidate: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let alignment = 8 << (scales - 1);
    let (mut w, mut h) = (
        width.next_multiple_of(alignment),
        height.next_multiple_of(alignment),
    );
    let padded = |sample: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
        (0..w * h)
            .map(|i| sample((i % w).min(width - 1), (i / w).min(height - 1)))
            .collect()
    };
    let (mut reference, mut candidate) = (padded(&reference), padded(&candidate));
    let mut deficits = Vec::with_capacity(scales);
    for scale in 0..scales {
        let mut sum = 0.0;
        for y in (0..h).step_by(8) {
            for x in (0..w).step_by(8) {
                let indices = || (y..y + 8).flat_map(|y| (x..x + 8).map(move |x| y * w + x));
                let a = indices().map(|i| reference[i]).sum::<f64>() / 64.0;
                let b = indices().map(|i| candidate[i]).sum::<f64>() / 64.0;
                let (mut variance_a, mut variance_b, mut covariance) = (0.0, 0.0, 0.0);
                for i in indices() {
                    let (da, db) = (reference[i] - a, candidate[i] - b);
                    variance_a += da * da / 64.0;
                    variance_b += db * db / 64.0;
                    covariance += da * db / 64.0;
                }
                let luminance = (2.0 * a * b + 1e-4) / (a * a + b * b + 1e-4);
                let structure = (2.0 * covariance + 9e-4) / (variance_a + variance_b + 9e-4);
                sum += 1.0 - luminance * structure;
            }
        }
        deficits.push(sum / (w * h / 64) as f64);
        if scale + 1 < scales {
            let halve = |plane: &[f64]| -> Vec<f64> {
                let half_width = w / 2;
                (0..w * h / 4)
                    .map(|i| {
                        let (x, y) = (i % half_width, i / half_width);
                        let top_left = 2 * (y * w + x);
                        (plane[top_left]
                            + plane[top_left + 1]
                            + plane[top_left + w]
                            + plane[top_left + w + 1])
                            / 4.0
                    })
                    .collect()
            };
            (reference, candidate) = (halve(&reference), halve(&candidate));
            (w, h) = (w / 2, h / 2);
        }
    }
    deficits
}

#[test]
fn mixed_strided_inputs_match_oracle_across_tiles_and_scales() {
    // Cross a tile boundary on each axis, with partial windows at the edges.
    let (w, h) = (263, 259);
    let (byte_stride, float_stride) = (w + 13, w + 29);
    let a = texture(w, h, 3);
    let b = texture(w, h, 7);
    let mut bytes = vec![0xff; (h - 1) * byte_stride + w];
    let mut floats = vec![f32::NAN; (h - 1) * float_stride + w];
    for y in 0..h {
        bytes[y * byte_stride..y * byte_stride + w].copy_from_slice(&a[y * w..(y + 1) * w]);
        for x in 0..w {
            floats[y * float_stride + x] = f32::from(b[y * w + x]) / 255.0;
        }
    }
    let reference = GrayImage::u8(&bytes, w, h, byte_stride).unwrap();
    let candidate = GrayImage::unit_f32(&floats, w, h, float_stride).unwrap();
    for scales in 1..=MAX_SCALES {
        let expected = oracle(
            w,
            h,
            scales,
            |x, y| f64::from(a[y * w + x]) / 255.0,
            |x, y| f64::from(floats[y * float_stride + x]),
        );
        let check = |actual: &[f64]| {
            assert_eq!(actual.len(), scales);
            for (scale, (&actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "{scales} scales, level {scale}: {actual} vs {expected}"
                );
            }
        };
        let prepared = PreparedReference::with_scales(reference, scales).unwrap();
        check(prepared.compare(candidate).unwrap().per_scale());
        if scales == DEFAULT_SCALES {
            check(
                msssim_lite::compare(reference, candidate)
                    .unwrap()
                    .per_scale(),
            );
            check(
                msssim_lite::compare(candidate, reference)
                    .unwrap()
                    .per_scale(),
            );
            let prepared = PreparedReference::new(candidate).unwrap();
            check(prepared.compare(reference).unwrap().per_scale());
        }
    }
}

#[test]
fn byte_reference_matches_oracle_for_sparse_and_maximum_errors() {
    let (w, h) = (263, 67);
    let (reference_stride, candidate_stride) = (w + 11, w + 23);
    let mut bytes = vec![0; (h - 1) * reference_stride + w];
    let mut sparse = vec![0x80; (h - 1) * candidate_stride + w];
    let mut black = sparse.clone();
    for y in 0..h {
        for x in 0..w {
            bytes[y * reference_stride + x] = 255;
            sparse[y * candidate_stride + x] = 255;
            black[y * candidate_stride + x] = 0;
        }
    }
    for (x, y) in [(16, 7), (255, 31), (w - 1, h - 1)] {
        sparse[y * candidate_stride + x] = 254;
    }
    let reference = GrayImage::u8(&bytes, w, h, reference_stride).unwrap();

    for scales in 1..=MAX_SCALES {
        let prepared = PreparedReference::with_scales(reference, scales).unwrap();
        assert_eq!(prepared.compare(reference).unwrap().deficit(), 0.0);
        for candidate in [&sparse, &black] {
            let expected = oracle(
                w,
                h,
                scales,
                |_, _| 1.0,
                |x, y| f64::from(candidate[y * candidate_stride + x]) / 255.0,
            );
            let candidate = GrayImage::u8(candidate, w, h, candidate_stride).unwrap();
            let check = |actual: &[f64]| {
                assert_eq!(actual.len(), scales);
                for (scale, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    // Coarse float levels round during box filtering.
                    let tolerance = if scale == 0 {
                        expected * 1e-6
                    } else {
                        expected * 1e-3 + 1e-12
                    };
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "{scales} scales, level {scale}: {actual} vs {expected}"
                    );
                }
            };
            check(prepared.compare(candidate).unwrap().per_scale());
            if scales == DEFAULT_SCALES {
                check(
                    msssim_lite::compare(reference, candidate)
                        .unwrap()
                        .per_scale(),
                );
            }
        }
    }
}

#[test]
fn tiny_difference_in_bright_window_has_a_positive_deficit() {
    let reference = [0.98; 64];
    let mut candidate = reference;
    candidate[0] += 1.0 / 131_072.0;
    let expected = oracle(
        8,
        8,
        1,
        |x, y| f64::from(reference[y * 8 + x]),
        |x, y| f64::from(candidate[y * 8 + x]),
    )[0];
    let prepared =
        PreparedReference::with_scales(GrayImage::unit_f32(&reference, 8, 8, 8).unwrap(), 1)
            .unwrap();
    let actual = prepared
        .compare(GrayImage::unit_f32(&candidate, 8, 8, 8).unwrap())
        .unwrap()
        .deficit();
    assert!(expected > 0.0);
    assert!(
        (actual - expected).abs() < expected * 1e-5,
        "{actual} vs {expected}"
    );
}

#[test]
fn nonfinite_samples_are_checked_through_the_last_visible_pixel() {
    let (w, h, stride) = (263, 67, 279);
    let mut floats = vec![f32::NAN; (h - 1) * stride + w];
    for row in floats.chunks_mut(stride) {
        row[..w].fill(0.0);
    }
    let bytes = vec![0; w * h];
    let reference = GrayImage::u8(&bytes, w, h, w).unwrap();
    let prepared = PreparedReference::new(reference).unwrap();
    let candidate = GrayImage::unit_f32(&floats, w, h, stride).unwrap();
    assert_eq!(prepared.compare(candidate).unwrap().deficit(), 0.0);
    assert_eq!(
        msssim_lite::compare(candidate, reference)
            .unwrap()
            .deficit(),
        0.0
    );

    // Check the end of a complete tile as well as the replicated edge tile.
    for (x, y) in [(255, 63), (w - 1, h - 1)] {
        floats[y * stride + x] = f32::INFINITY;
        let candidate = GrayImage::unit_f32(&floats, w, h, stride).unwrap();
        assert_eq!(prepared.compare(candidate), Err(Error::NonFiniteSample));
        assert_eq!(
            msssim_lite::compare(reference, candidate),
            Err(Error::NonFiniteSample)
        );
        assert_eq!(
            PreparedReference::new(candidate).err(),
            Some(Error::NonFiniteSample)
        );
        floats[y * stride + x] = 0.0;
    }
}
