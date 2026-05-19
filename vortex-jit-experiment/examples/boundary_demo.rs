// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Demo: show the kernel boundary by running the same operation three ways.
//!
//! Run:
//!     cargo run --release -p vortex-jit-experiment --bin boundary-demo

use std::time::Instant;

use vortex_jit_experiment::CHUNK_SIZE;
use vortex_jit_experiment::MASK_WORDS;
use vortex_jit_experiment::composed;
use vortex_jit_experiment::fused;
use vortex_jit_experiment::jit;
use vortex_jit_experiment::pack::pack_chunk;

fn main() {
    let bit_width: u32 = 11;
    let max = (1u32 << bit_width) - 1;
    let threshold = max / 2; // ~50% selectivity

    println!("=== kernel boundary demo ===");
    println!("bit_width = {bit_width}, threshold k = {threshold}");
    println!("chunk size = {CHUNK_SIZE} elements");
    println!();

    // Build a deterministic input chunk in [0, 2^bit_width).
    let mut values = [0u32; CHUNK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ((i as u32).wrapping_mul(0x9E37_79B1)) & max;
    }
    // Pack with `REQUIRED_PAD_WORDS` of trailing zero padding so the JIT's
    // unconditional 8-byte load never falls off the end.
    let mut packed = pack_chunk(&values, bit_width);
    packed.extend(std::iter::repeat_n(0u32, jit::REQUIRED_PAD_WORDS));
    println!(
        "packed footprint: {} value words + {} pad = {} u32 ({} bytes)",
        jit::n_value_words(bit_width),
        jit::REQUIRED_PAD_WORDS,
        packed.len(),
        packed.len() * 4,
    );
    println!();

    // -------- 1. Composed (today's vx kernel boundary, with temp buffer) --------
    let mut mask_composed = [0u64; MASK_WORDS];
    composed::unpack_then_compare(&packed, bit_width, threshold, &mut mask_composed);

    // -------- 2. Hand-fused (what a perfect JIT would emit) --------
    let mut mask_fused = [0u64; MASK_WORDS];
    fused::unpack_compare_fused(&packed, bit_width, threshold, &mut mask_fused);

    // -------- 3. JIT --------
    let t0 = Instant::now();
    let kernel = jit::compile(bit_width).expect("JIT compile failed");
    let jit_compile_time = t0.elapsed();
    let mut mask_jit = [0u64; MASK_WORDS];
    // SAFETY: padded above to REQUIRED_PAD_WORDS.
    unsafe { kernel.run(&packed, threshold, &mut mask_jit) };

    // -------- Equivalence --------
    assert_eq!(mask_composed, mask_fused, "composed vs fused disagree");
    assert_eq!(mask_composed, mask_jit, "composed vs JIT disagree");
    let popcount: u32 = mask_composed.iter().map(|w| w.count_ones()).sum();
    println!("all three paths agree: {popcount} / {CHUNK_SIZE} elements pass (> {threshold})",);
    println!();

    // -------- Show the JIT'd IR --------
    println!("--- Cranelift IR (pre-lowering) ---");
    for line in kernel.ir().lines() {
        println!("    {line}");
    }
    println!();

    // -------- Timing --------
    let iters = 200_000u32;
    let bench = |name: &str, mut f: Box<dyn FnMut()>| {
        // Warm up.
        for _ in 0..1000 {
            f();
        }
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = t.elapsed();
        let ns_per_chunk = elapsed.as_nanos() as f64 / iters as f64;
        let ns_per_elem = ns_per_chunk / CHUNK_SIZE as f64;
        println!("  {name:20}  {ns_per_chunk:>7.1} ns/chunk   {ns_per_elem:>5.2} ns/elem",);
    };
    println!("--- Timing ({iters} iters, {CHUNK_SIZE} elements/chunk) ---");
    {
        let mut mask = [0u64; MASK_WORDS];
        let packed_ref = &packed;
        bench(
            "composed",
            Box::new(move || {
                composed::unpack_then_compare(packed_ref, bit_width, threshold, &mut mask);
            }),
        );
    }
    {
        let mut mask = [0u64; MASK_WORDS];
        let packed_ref = &packed;
        bench(
            "hand-fused",
            Box::new(move || {
                fused::unpack_compare_fused(packed_ref, bit_width, threshold, &mut mask);
            }),
        );
    }
    {
        let mut mask = [0u64; MASK_WORDS];
        let packed_ref = &packed;
        let kernel_ref = &kernel;
        bench(
            "JIT-fused",
            Box::new(move || {
                // SAFETY: padded above.
                unsafe { kernel_ref.run(packed_ref, threshold, &mut mask) };
            }),
        );
    }
    println!();
    println!("JIT compile latency for this kernel: {jit_compile_time:?}");
}
