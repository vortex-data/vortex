// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CPU-feature reporting and a fairness guard.
//!
//! Every comparison in this harness lives in one binary that links Arrow and
//! our kernels together, so they are necessarily compiled with the *same*
//! `target-feature` set and run on the *same* CPU. That is the only way to make
//! an "is the split faster than Arrow?" claim honest: if our kernel used
//! AVX-512 while Arrow's codegen was restricted to SSE2, the comparison would
//! be meaningless.
//!
//! `report()` prints both the compile-time feature set (what the optimizer was
//! allowed to use for *both* sides, including Arrow's autovectorized loops) and
//! the run-time detected set. `require_simd()` refuses to report numbers when
//! the binary was built without the wide ISA, so we never accidentally pit an
//! AVX-512 kernel against a scalar Arrow build.

/// x86-64 features we care about for decimal work, as (name, compiled-in,
/// detected-at-runtime) triples.
pub fn features() -> Vec<(&'static str, bool, bool)> {
    #[cfg(target_arch = "x86_64")]
    {
        vec![
            ("avx2", cfg!(target_feature = "avx2"), std::arch::is_x86_feature_detected!("avx2")),
            ("avx512f", cfg!(target_feature = "avx512f"), std::arch::is_x86_feature_detected!("avx512f")),
            ("avx512bw", cfg!(target_feature = "avx512bw"), std::arch::is_x86_feature_detected!("avx512bw")),
            ("avx512dq", cfg!(target_feature = "avx512dq"), std::arch::is_x86_feature_detected!("avx512dq")),
            ("avx512vl", cfg!(target_feature = "avx512vl"), std::arch::is_x86_feature_detected!("avx512vl")),
            ("avx512cd", cfg!(target_feature = "avx512cd"), std::arch::is_x86_feature_detected!("avx512cd")),
        ]
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

/// True if the binary was compiled with AVX-512F available to the optimizer,
/// i.e. both Arrow and our kernels could use it.
pub fn compiled_with_avx512() -> bool {
    cfg!(target_feature = "avx512f")
}

/// Print the compile-time vs run-time feature matrix as a Markdown table.
pub fn report() {
    println!("## CPU feature parity (Arrow and Vortex kernels share these)\n");
    println!("| feature | compiled-in | detected |");
    println!("|---|:---:|:---:|");
    for (name, compiled, detected) in features() {
        println!("| {name} | {} | {} |", yes_no(compiled), yes_no(detected));
    }
    if !compiled_with_avx512() {
        println!(
            "\n> WARNING: built WITHOUT avx512f. Both sides fall back to narrower \
             ISA, so absolute numbers differ from a native build. Rebuild with \
             `RUSTFLAGS=\"-C target-cpu=native\"` (or `-C target-feature=+avx512f,...`) \
             for the intended comparison."
        );
    } else {
        println!(
            "\n> Both Arrow's autovectorized kernels and the split kernels were \
             compiled with the same feature set above; the comparison is apples-to-apples."
        );
    }
    println!();
}

/// Refuse to produce comparison numbers unless the wide ISA was compiled in, so
/// a fair comparison is enforced rather than assumed. Returns whether AVX-512 is
/// both compiled-in and detected.
pub fn require_simd() -> bool {
    let detected = features()
        .iter()
        .find(|(n, _, _)| *n == "avx512f")
        .is_some_and(|&(_, _, d)| d);
    compiled_with_avx512() && detected
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}
