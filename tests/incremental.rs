use msssim_lite::{
    Checkpoint, DEFAULT_SCALES, Error, GrayImage, IncrementalState, MAX_SCALES, PreparedReference,
    Rect, Score,
};

fn texture(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| ((i * 47 + (i / w) * 13 + (i % w) * (i / w)) % 251) as u8)
        .collect()
}

fn bytes(samples: &[u8], w: usize, h: usize) -> GrayImage<'_> {
    GrayImage::u8(samples, w, h, w).unwrap()
}

fn floats(samples: &[f32], w: usize, h: usize, stride: usize) -> GrayImage<'_> {
    GrayImage::unit_f32(samples, w, h, stride).unwrap()
}

fn check_score(actual: Score, expected: Score) {
    assert_eq!(actual.per_scale().len(), expected.per_scale().len());
    for (scale, (&actual, &expected)) in actual
        .per_scale()
        .iter()
        .zip(expected.per_scale())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-7,
            "scale {scale}: {actual} vs {expected}"
        );
    }
}

#[test]
fn byte_updates_match_full_comparisons_and_reuse_checkpoints() {
    let (w, h) = (269, 67);
    let stride = w + 13;
    let source = texture(w, h);

    for scales in [1, DEFAULT_SCALES, MAX_SCALES] {
        let prepared = PreparedReference::with_scales(bytes(&source, w, h), scales).unwrap();
        let mut accepted: Vec<_> = source.iter().map(|&v| v & !3).collect();
        let mut state = prepared.incremental(bytes(&accepted, w, h)).unwrap();
        assert_eq!(
            (state.width(), state.height(), state.scales()),
            (w, h, scales)
        );
        check_score(
            state.score(),
            prepared.compare(bytes(&accepted, w, h)).unwrap(),
        );

        let mut checkpoint = Checkpoint::default();
        for rect in [
            Rect::new(7, 6, 11, 10),
            Rect::new(9, 8, 35, 34),
            Rect::new(253, 62, w + 7, h + 4),
            Rect::new(w - 1, h - 1, w, h),
        ] {
            // Only the rectangle is populated. Padding and the rest of the
            // input deliberately differ from the accepted candidate.
            let mut staged = vec![0xef; (h - 1) * stride + w];
            let mut trial = accepted.clone();
            for y in rect.y0..rect.y1.min(h) {
                for x in rect.x0..rect.x1.min(w) {
                    let pixel = trial[y * w + x].wrapping_add(23);
                    trial[y * w + x] = pixel;
                    staged[y * stride + x] = pixel;
                }
            }
            let input = GrayImage::u8(&staged, w, h, stride).unwrap();
            let before = state.score();
            let expected = prepared.compare(bytes(&trial, w, h)).unwrap();
            state
                .update_checkpointed(input, rect, &mut checkpoint)
                .unwrap();
            check_score(state.score(), expected);
            let retained = (state.allocated_bytes(), checkpoint.allocated_bytes());
            state.rollback(&checkpoint);
            assert_eq!(state.score(), before);

            state
                .update_checkpointed(input, rect, &mut checkpoint)
                .unwrap();
            assert_eq!(
                (state.allocated_bytes(), checkpoint.allocated_bytes()),
                retained
            );
            state.rollback(&checkpoint);
            assert_eq!(state.score(), before);

            state.update(input, rect).unwrap();
            check_score(state.score(), expected);
            accepted = trial;
        }
    }
}

#[test]
fn update_and_rollback_across_tile_tree_depths() {
    let (w, h) = (41, 21);
    let reference = texture(w, h);
    let prepared = PreparedReference::with_scales(bytes(&reference, w, h), 1).unwrap();
    let initial: Vec<_> = reference.iter().map(|&v| v & !3).collect();
    let mut state = prepared.incremental(bytes(&initial, w, h)).unwrap();
    let baseline = state.score();

    // These tiles include the last shallow leaf and the first deeper leaves.
    let rect = Rect::new(10, 16, w, h);
    let mut trial = initial;
    for row in trial.chunks_exact_mut(w).skip(rect.y0) {
        row[rect.x0..].fill(127);
    }
    let mut checkpoint = Checkpoint::default();
    let trial = bytes(&trial, w, h);
    state
        .update_checkpointed(trial, rect, &mut checkpoint)
        .unwrap();
    check_score(state.score(), prepared.compare(trial).unwrap());
    state.rollback(&checkpoint);
    assert_eq!(state.score(), baseline);
}

#[test]
fn float_update_reads_only_its_rectangle_and_errors_are_transactional() {
    let (w, h) = (73, 51);
    let stride = w + 9;
    let source = texture(w, h);
    let prepared = PreparedReference::new(GrayImage::u8(&source, w, h, w).unwrap()).unwrap();
    let initial: Vec<_> = source.iter().map(|&v| f32::from(v) / 256.0).collect();
    let mut state = IncrementalState::new(&prepared, floats(&initial, w, h, w)).unwrap();
    let baseline = state.score();
    check_score(
        baseline,
        prepared.compare(floats(&initial, w, h, w)).unwrap(),
    );

    let mut staged = vec![f32::NAN; (h - 1) * stride + w];
    let mut expected = initial.clone();
    let rect = Rect::new(9, 9, 11, 11);
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            staged[y * stride + x] = 0.25;
            expected[y * w + x] = 0.25;
        }
    }
    let mut checkpoint = Checkpoint::default();
    state
        .update_checkpointed(floats(&staged, w, h, stride), rect, &mut checkpoint)
        .unwrap();
    let accepted = state.score();
    check_score(
        accepted,
        prepared.compare(floats(&expected, w, h, w)).unwrap(),
    );

    let bad_rect = Rect::new(12, 9, 15, 11);
    staged[9 * stride + 12..9 * stride + 15].fill(0.9);
    staged[10 * stride + 12..10 * stride + 14].fill(0.9);
    staged[10 * stride + 14] = f32::INFINITY;
    assert_eq!(
        state.update_checkpointed(floats(&staged, w, h, stride), bad_rect, &mut checkpoint),
        Err(Error::NonFiniteSample)
    );
    assert_eq!(
        state.update_checkpointed(
            GrayImage::u8(&source, w, h, w).unwrap(),
            rect,
            &mut checkpoint
        ),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        state.update_checkpointed(
            GrayImage::unit_f32(&initial, w - 1, h, w).unwrap(),
            rect,
            &mut checkpoint,
        ),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(state.score(), accepted);

    // Re-score the same window to expose partial writes from the failed update.
    let mut trial = Checkpoint::default();
    staged[11 * stride + 11] = 0.75;
    expected[11 * w + 11] = 0.75;
    state
        .update_checkpointed(
            floats(&staged, w, h, stride),
            Rect::new(11, 11, 12, 12),
            &mut trial,
        )
        .unwrap();
    check_score(
        state.score(),
        prepared.compare(floats(&expected, w, h, w)).unwrap(),
    );
    state.rollback(&trial);
    assert_eq!(state.score(), accepted);
    state.rollback(&checkpoint);
    assert_eq!(state.score(), baseline);
}

#[test]
fn compact_float_patch_at_the_edge_matches_full_scoring() {
    let (w, h) = (73, 51);
    let source = texture(w, h);
    let prepared = PreparedReference::new(bytes(&source, w, h)).unwrap();
    let initial: Vec<_> = source.iter().map(|&v| f32::from(v) / 256.0).collect();
    let mut state = prepared.incremental(floats(&initial, w, h, w)).unwrap();
    let baseline = state.score();
    let (patch_w, patch_h, stride) = (7, 5, 18);
    let (x, y) = (w - patch_w, h - patch_h);
    let mut patch = vec![f32::NAN; (patch_h - 1) * stride + patch_w];
    let mut expected = initial.clone();
    for row in 0..patch_h {
        for col in 0..patch_w {
            let sample = (row * patch_w + col) as f32 / 64.0;
            patch[row * stride + col] = sample;
            expected[(y + row) * w + x + col] = sample;
        }
    }
    let mut checkpoint = Checkpoint::default();
    state
        .update_patch_checkpointed(
            floats(&patch, patch_w, patch_h, stride),
            x,
            y,
            &mut checkpoint,
        )
        .unwrap();
    let patched = state.score();
    check_score(
        patched,
        prepared.compare(floats(&expected, w, h, w)).unwrap(),
    );

    *patch.last_mut().unwrap() = f32::NAN;
    assert_eq!(
        state.update_patch_checkpointed(
            floats(&patch, patch_w, patch_h, stride),
            x,
            y,
            &mut checkpoint,
        ),
        Err(Error::NonFiniteSample)
    );
    assert_eq!(state.score(), patched);
    state.rollback(&checkpoint);
    assert_eq!(state.score(), baseline);
}

#[test]
fn compact_byte_patch_rejects_invalid_bounds_and_format() {
    let (w, h) = (37, 35);
    let source = texture(w, h);
    let prepared = PreparedReference::new(bytes(&source, w, h)).unwrap();
    let mut state = prepared.incremental(bytes(&source, w, h)).unwrap();
    let baseline = state.score();
    let patch = GrayImage::u8(&[25, 43, 77, 255, 63, 38, 64], 3, 2, 4).unwrap();
    let (x, y) = (17, 9);
    let mut expected = source.clone();
    expected[y * w + x..][..3].copy_from_slice(&[25, 43, 77]);
    expected[(y + 1) * w + x..][..3].copy_from_slice(&[63, 38, 64]);
    let mut checkpoint = Checkpoint::default();
    state
        .update_patch_checkpointed(patch, x, y, &mut checkpoint)
        .unwrap();
    let patched = state.score();
    check_score(patched, prepared.compare(bytes(&expected, w, h)).unwrap());

    for (x, y) in [(w - 2, 0), (0, h - 1), (usize::MAX, 0), (0, usize::MAX)] {
        assert_eq!(
            state.update_patch_checkpointed(patch, x, y, &mut checkpoint),
            Err(Error::DimensionMismatch)
        );
    }
    assert_eq!(
        state.update_patch_checkpointed(floats(&[0.0; 6], 3, 2, 3), x, y, &mut checkpoint),
        Err(Error::InvalidLayout)
    );
    assert_eq!(state.score(), patched);
    state.rollback(&checkpoint);
    assert_eq!(state.score(), baseline);
}

#[test]
fn checkpoints_only_restore_the_matching_state_and_revision() {
    let (w, h) = (37, 35);
    let source = texture(w, h);
    let prepared = PreparedReference::new(bytes(&source, w, h)).unwrap();
    let mut state = prepared.incremental(bytes(&source, w, h)).unwrap();
    let mut other = prepared.incremental(bytes(&source, w, h)).unwrap();
    drop(prepared);
    let initial = state.score();
    let trial = vec![127; w * h];
    let first = Rect::new(3, 3, 8, 8);
    let second = Rect::new(31, 30, 37, 35);
    let mut outer = Checkpoint::default();
    let mut inner = Checkpoint::default();

    state
        .update_checkpointed(bytes(&trial, w, h), first, &mut outer)
        .unwrap();
    let after_first = state.score();
    other.update(bytes(&trial, w, h), first).unwrap();
    let other_score = other.score();
    other.rollback(&outer);
    assert_eq!(other.score(), other_score);

    state
        .update_checkpointed(bytes(&trial, w, h), second, &mut inner)
        .unwrap();
    let after_second = state.score();
    state.rollback(&outer);
    assert_eq!(state.score(), after_second);
    state.rollback(&inner);
    assert_eq!(state.score(), after_first);
    state.rollback(&inner);
    assert_eq!(state.score(), after_first);

    // A new update after rollback must use a fresh revision.
    state.update(bytes(&trial, w, h), second).unwrap();
    let after_commit = state.score();
    check_score(after_commit, after_second);
    state.rollback(&inner);
    assert_eq!(state.score(), after_commit);
    state.rollback(&outer);
    assert_eq!(state.score(), after_commit);
    assert_ne!(state.score(), initial);

    state
        .update_checkpointed(bytes(&source, w, h), second, &mut inner)
        .unwrap();
    let before_noop = state.score();
    state
        .update_checkpointed(
            bytes(&trial, w, h),
            Rect::new(w + 1, 0, w + 3, h),
            &mut inner,
        )
        .unwrap();
    assert_eq!(state.score(), before_noop);
    state.rollback(&inner);
    assert_eq!(state.score(), before_noop);

    state
        .update(bytes(&source, w, h), Rect::new(0, 0, w, h))
        .unwrap();
    assert_eq!(state.score().deficit(), 0.0);
}

#[test]
fn unchanged_updates_advance_and_restore_checkpoint_revisions() {
    let (w, h) = (37, 35);
    let source = texture(w, h);
    let initial = bytes(&source, w, h);
    let prepared = PreparedReference::new(initial).unwrap();
    let mut state = prepared.incremental(initial).unwrap();
    let mut outer = Checkpoint::default();
    let mut unchanged = Checkpoint::default();
    let (x, y) = (w - 2, h - 2);
    let patch_samples = [91, 123, 0, 215, 46];
    let patch = GrayImage::u8(&patch_samples, 2, 2, 3).unwrap();

    state
        .update_patch_checkpointed(patch, x, y, &mut outer)
        .unwrap();
    let changed = state.score();
    state
        .update_patch_checkpointed(patch, x, y, &mut unchanged)
        .unwrap();
    assert_eq!(unchanged.allocated_bytes(), 0);
    state.rollback(&outer);
    assert_eq!(state.score(), changed);
    state.rollback(&unchanged);
    state.rollback(&outer);
    assert_eq!(state.score().deficit(), 0.0);

    state
        .update_patch_checkpointed(patch, x, y, &mut outer)
        .unwrap();
    state
        .update_patch_checkpointed(patch, x, y, &mut outer)
        .unwrap();
    state.rollback(&outer);
    assert_eq!(state.score(), changed);

    // An unchanged committed update also advances the revision.
    state
        .update_checkpointed(initial, Rect::new(x, y, w, h), &mut outer)
        .unwrap();
    state.update(initial, Rect::new(x, y, w, h)).unwrap();
    state.rollback(&outer);
    assert_eq!(state.score().deficit(), 0.0);
}

#[test]
fn sparse_strided_update_checkpoints_only_the_changed_bounds() {
    let (w, h) = (269, 137);
    let source = texture(w, h);
    let initial = bytes(&source, w, h);
    let prepared = PreparedReference::new(initial).unwrap();
    let mut state = prepared.incremental(initial).unwrap();
    let before = state.score();
    let (x, y) = (249, 118);
    let (pw, ph, stride) = (w - x, h - y, 29);
    let mut patch = vec![255; (ph - 1) * stride + pw];
    for (row, dst) in patch.chunks_mut(stride).enumerate() {
        dst[..pw].copy_from_slice(&source[(y + row) * w + x..][..pw]);
    }
    let mut expected = source.clone();
    let edits = [(pw - 3, ph - 4), (pw - 2, ph - 3), (pw - 4, ph - 1)];
    for (px, py) in edits {
        patch[py * stride + px] ^= 123;
        expected[(y + py) * w + x + px] = patch[py * stride + px];
    }
    let mut expected_state = prepared.incremental(initial).unwrap();
    let mut expected_checkpoint = Checkpoint::default();
    expected_state
        .update_checkpointed(
            bytes(&expected, w, h),
            Rect::new(w - 4, h - 4, w - 1, h),
            &mut expected_checkpoint,
        )
        .unwrap();
    let mut checkpoint = Checkpoint::default();
    state
        .update_patch_checkpointed(
            GrayImage::u8(&patch, pw, ph, stride).unwrap(),
            x,
            y,
            &mut checkpoint,
        )
        .unwrap();
    assert_eq!(state.score(), expected_state.score());
    // The 3×4 bounds fit in a single tile. The original patch spans four.
    assert_eq!(
        checkpoint.allocated_bytes(),
        expected_checkpoint.allocated_bytes()
    );
    state.rollback(&checkpoint);
    assert_eq!(state.score(), before);
}
