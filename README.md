# msssim-lite

A dependency-free, fast multi-scale SSIM on single-channel images, meant as a
cheap stand-in for a perceptual metric inside encoder decisions.

## API

```rust
use msssim_lite::{GrayImage, PreparedReference};

let prepared = PreparedReference::new(GrayImage::unit_f32(&reference, width, height, width)?)?;
let a = prepared.compare(GrayImage::unit_f32(&candidate_a, width, height, width)?)?;
let b = prepared.compare(GrayImage::unit_f32(&candidate_b, width, height, width)?)?;
let a_is_closer = a.deficit() < b.deficit();
```

```rust
use msssim_lite::{Checkpoint, Rect};

let mut state = prepared.incremental(current)?;
let mut checkpoint = Checkpoint::default();
state.update_checkpointed(trial, Rect::new(x0, y0, x1, y1), &mut checkpoint)?;
let trial_score = state.score();
state.rollback(&checkpoint);
```

## License

Licensed under either of

- BSD-3-Clause ([LICENSE.md](LICENSE.md))
- Apache License, Version 2.0 ([LICENSE-APACHE.md](LICENSE-APACHE.md))

at your option.
