// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected as detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected as detected;
use std::collections::BTreeSet;
use std::env;
use std::env::consts::ARCH;

const EXPECT_VAR: &str = "VORTEX_EXPECT_CPU_FEATURES";

/// The features that select a kernel somewhere in Vortex, with their runtime detection.
#[cfg(target_arch = "x86_64")]
fn known_features() -> Vec<(&'static str, bool)> {
    vec![
        ("sse2", detected!("sse2")),
        ("popcnt", detected!("popcnt")),
        ("bmi1", detected!("bmi1")),
        ("bmi2", detected!("bmi2")),
        ("avx", detected!("avx")),
        ("avx2", detected!("avx2")),
        ("avx512f", detected!("avx512f")),
        ("avx512bw", detected!("avx512bw")),
        ("avx512vbmi2", detected!("avx512vbmi2")),
        ("avx512vpopcntdq", detected!("avx512vpopcntdq")),
    ]
}

/// The features that select a kernel somewhere in Vortex, with their runtime detection.
#[cfg(target_arch = "aarch64")]
fn known_features() -> Vec<(&'static str, bool)> {
    vec![
        ("neon", detected!("neon")),
        ("sve", detected!("sve")),
        ("sve2", detected!("sve2")),
        ("sve2-bitperm", detected!("sve2-bitperm")),
    ]
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn known_features() -> Vec<(&'static str, bool)> {
    Vec::new()
}

#[test]
fn expected_cpu_features_are_detected() {
    let known = known_features();
    for (name, got) in &known {
        println!(
            "{ARCH} {name}: {}",
            if *got { "detected" } else { "not detected" }
        );
    }

    let Ok(expected) = env::var(EXPECT_VAR) else {
        println!("{EXPECT_VAR} is not set, skipping the check");
        return;
    };
    println!("{EXPECT_VAR}={expected}");
    let expected: BTreeSet<&str> = expected
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();

    let mut failures: Vec<String> = expected
        .iter()
        .filter(|feature| !known.iter().any(|(name, _)| name == *feature))
        .map(|feature| format!("{feature}: unknown feature for {ARCH}"))
        .collect();
    for (name, got) in &known {
        let want = expected.contains(name);
        if want != *got {
            failures.push(format!("{name}: expected {want}, detected {got}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{EXPECT_VAR} does not match this CPU:\n{}",
        failures.join("\n")
    );
}
