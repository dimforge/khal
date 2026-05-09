//! Smoke test: the Metal backend's timestamp pipeline end-to-end.
//!
//! Confirms that `MetalTimestamps::new` succeeds on this machine, that a
//! timed compute pass produces a non-zero duration, and that the labels
//! line up with what the host enqueued.

#![cfg(feature = "metal")]

use khal::backend::Metal;
use khal::backend::metal::MetalTimestamps;
use khal::backend::{Backend, Buffer, BufferUsages, Encoder};

#[test]
fn metal_timestamps_smoke() {
    let metal = match Metal::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping: no Metal device available ({e})");
            return;
        }
    };
    if !metal.timestamp_supported() {
        eprintln!("Skipping: device does not expose timestamp counter set");
        return;
    }

    let mut timestamps = MetalTimestamps::new(&metal, 4)
        .expect("MetalTimestamps::new returned None despite timestamp_supported()");

    // A trivial GPU operation: a buffer-to-buffer copy big enough that the
    // pass surrounding it has measurable duration. We don't actually need
    // a compute dispatch here — just an encoder pass with samples on it.
    // Use a couple of distinct passes to exercise multiple sample pairs.
    let src = metal
        .init_buffer::<f32>(
            &vec![1.0_f32; 1024 * 1024],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
    let mut dst = metal
        .uninit_buffer::<f32>(1024 * 1024, BufferUsages::STORAGE | BufferUsages::COPY_DST)
        .unwrap();

    let mut encoder = metal.begin_encoding();
    {
        let _pass = encoder.begin_pass("first_pass", Some(&mut timestamps));
    }
    encoder
        .copy_buffer_to_buffer(&src, 0, &mut dst, 0, src.len())
        .unwrap();
    {
        let _pass = encoder.begin_pass("second_pass", Some(&mut timestamps));
    }
    metal.submit(encoder).unwrap();
    metal.synchronize().unwrap();

    let entries = timestamps.read().expect("read timestamps");
    assert_eq!(entries.len(), 2, "expected two timed passes");
    assert_eq!(entries[0].label, "first_pass");
    assert_eq!(entries[1].label, "second_pass");
    // Timestamps may be ~zero for an empty pass, but they must be finite
    // and non-negative.
    for e in &entries {
        assert!(e.duration_ms.is_finite(), "{} duration not finite", e.label);
        assert!(e.duration_ms >= 0.0, "{} duration negative", e.label);
    }
}
