// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast property smoke tests bridging `hegeltest` to `vortex-fuzz` harnesses.
//!
//! Draws raw byte vectors from `hegel::generators`, decodes each fuzz target's `Arbitrary` input
//! via `arbitrary::Unstructured`, and exercises the fuzz oracles under `cargo test` without
//! waiting for the scheduled libFuzzer workflow.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use hegel::HealthCheck;
use hegel::TestCase;
use hegel::generators as gs;
use vortex_error::vortex_panic;

use crate::FuzzArrayAction;
use crate::FuzzCompressRoundtrip;
use crate::FuzzRowEncode;
#[cfg(feature = "native")]
use crate::FuzzWireSafety;
use crate::run_compress_roundtrip;
use crate::run_fuzz_action;
use crate::run_row_encode;
#[cfg(feature = "native")]
use crate::run_wire_safety;

/// Maximum raw byte-vector length drawn per smoke case to keep test execution fast.
const MAX_SMOKE_BYTES: usize = 512;

fn draw_arbitrary<T>(tc: &TestCase, min_bytes: usize, max_bytes: usize) -> T
where
    T: for<'a> Arbitrary<'a>,
{
    let bytes = tc.draw(gs::binary().min_size(min_bytes).max_size(max_bytes));
    let mut unstructured = Unstructured::new(&bytes);
    match T::arbitrary(&mut unstructured) {
        Ok(value) => value,
        Err(_) => tc.reject(),
    }
}

#[hegel::test(
    test_cases = 50,
    print_blob = true,
    suppress_health_check = [HealthCheck::FilterTooMuch, HealthCheck::TooSlow]
)]
fn test_array_ops_smoke(tc: TestCase) {
    let action: FuzzArrayAction = draw_arbitrary(&tc, 16, MAX_SMOKE_BYTES);
    if let Err(err) = run_fuzz_action(action) {
        vortex_panic!("array_ops smoke test failed: {err}");
    }
}

#[hegel::test(
    test_cases = 50,
    print_blob = true,
    suppress_health_check = [HealthCheck::FilterTooMuch, HealthCheck::TooSlow]
)]
fn test_compress_roundtrip_smoke(tc: TestCase) {
    let case: FuzzCompressRoundtrip = draw_arbitrary(&tc, 16, MAX_SMOKE_BYTES);
    if let Err(err) = run_compress_roundtrip(case) {
        vortex_panic!("compress_roundtrip smoke test failed: {err}");
    }
}

#[hegel::test(
    test_cases = 50,
    print_blob = true,
    suppress_health_check = [HealthCheck::FilterTooMuch, HealthCheck::TooSlow]
)]
fn test_row_encode_smoke(tc: TestCase) {
    let case: FuzzRowEncode = draw_arbitrary(&tc, 16, MAX_SMOKE_BYTES);
    if let Err(err) = run_row_encode(case) {
        vortex_panic!("row_encode smoke test failed: {err}");
    }
}

#[cfg(feature = "native")]
#[hegel::test(
    test_cases = 50,
    print_blob = true,
    suppress_health_check = [HealthCheck::FilterTooMuch, HealthCheck::TooSlow]
)]
fn test_wire_safety_smoke(tc: TestCase) {
    let case: FuzzWireSafety = draw_arbitrary(&tc, 1, MAX_SMOKE_BYTES);
    run_wire_safety(case);
}
