use msssim_lite::{Checkpoint, Error, GrayImage, MAX_SCALES, PreparedReference, Rect};

#[test]
fn cached_scale_count_does_not_change_scores() {
    let (width, height) = (267, 33);
    let bytes: Vec<_> = (0..width * height)
        .map(|i| ((i * 17 + (i / width) * 31) % 256) as u8)
        .collect();
    let mut edited = bytes.clone();
    let edit = (height - 1) * width + width - 1;
    edited[edit] ^= 0x55;
    let floats = |samples: &[u8]| {
        samples
            .iter()
            .map(|&v| f32::from(v) / 255.0)
            .collect::<Vec<_>>()
    };
    let original_floats = floats(&bytes);
    let edited_floats = floats(&edited);
    let original_views = [
        GrayImage::u8(&bytes, width, height, width).unwrap(),
        GrayImage::unit_f32(&original_floats, width, height, width).unwrap(),
    ];
    let edited_views = [
        GrayImage::u8(&edited, width, height, width).unwrap(),
        GrayImage::unit_f32(&edited_floats, width, height, width).unwrap(),
    ];
    let patches = [
        GrayImage::u8(&edited[edit..], 1, 1, 1).unwrap(),
        GrayImage::unit_f32(&edited_floats[edit..], 1, 1, 1).unwrap(),
    ];

    for (format, original) in original_views.into_iter().enumerate() {
        for scales in 1..=MAX_SCALES {
            let full = PreparedReference::with_scales(original, scales).unwrap();
            for cached_scales in 1..scales {
                let compact =
                    PreparedReference::with_cached_scales(original, scales, cached_scales).unwrap();
                assert_eq!(compact.scales(), scales);
                assert_eq!(compact.cached_scales(), cached_scales);
                for edited in edited_views {
                    assert_eq!(compact.compare(edited), full.compare(edited));
                }
                assert_eq!(
                    compact.compare_patch(patches[format], width - 1, height - 1),
                    full.compare_patch(patches[format], width - 1, height - 1)
                );

                let mut compact_state = compact.incremental(original).unwrap();
                let mut full_state = full.incremental(original).unwrap();
                let mut compact_checkpoint = Checkpoint::default();
                let mut full_checkpoint = Checkpoint::default();
                let rect = Rect::new(width - 1, height - 1, width, height);
                compact_state
                    .update_checkpointed(edited_views[format], rect, &mut compact_checkpoint)
                    .unwrap();
                full_state
                    .update_checkpointed(edited_views[format], rect, &mut full_checkpoint)
                    .unwrap();
                assert_eq!(compact_state.score(), full_state.score());
                compact_state.rollback(&compact_checkpoint);
                full_state.rollback(&full_checkpoint);
                assert_eq!(compact_state.score(), full_state.score());
            }
        }
    }
}

#[test]
fn invalid_scale_counts_are_rejected() {
    let image = GrayImage::u8(&[0], 1, 1, 1).unwrap();
    for (scales, cached) in [(0, 0), (3, 0), (1, 2), (MAX_SCALES + 1, 1)] {
        assert_eq!(
            PreparedReference::with_cached_scales(image, scales, cached).err(),
            Some(Error::InvalidScales)
        );
    }
}
