// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Smoke test that drives the FSST roundtrip fuzz harness against a few
//! thousand pseudo-random inputs without needing libfuzzer. Useful for local
//! verification and for CI on non-libfuzzer platforms.

#![cfg(feature = "native")]

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_fuzz::FuzzFsstRoundtrip;
use vortex_fuzz::run_fsst_roundtrip_fuzz;

fn pseudo_random_bytes(seed: u64, len: usize) -> Vec<u8> {
    // Tiny LCG; enough entropy for fuzz-style input generation.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xDEAD_BEEF);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push((state >> 33) as u8);
    }
    out
}

#[test]
fn fuzz_fsst_roundtrip_many() {
    let iterations: usize = std::env::var("FSST_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000);

    let mut kept = 0usize;
    let mut rejected = 0usize;

    for seed in 0..iterations as u64 {
        // Generate a fuzz blob, then feed it into `Arbitrary` to build the
        // FuzzFsstRoundtrip input. This matches the libfuzzer code path.
        let blob = pseudo_random_bytes(seed, 4096 + (seed as usize % 2048));
        let mut u = Unstructured::new(&blob);
        let Ok(input) = FuzzFsstRoundtrip::arbitrary(&mut u) else {
            rejected += 1;
            continue;
        };

        match run_fsst_roundtrip_fuzz(input) {
            Ok(true) => kept += 1,
            Ok(false) => rejected += 1,
            Err(e) => panic!("seed {seed}: {e}"),
        }
    }

    eprintln!(
        "fsst_roundtrip smoke: {kept} kept / {rejected} rejected over {iterations} iterations"
    );
    assert!(kept > 0, "expected at least one kept iteration");
}
