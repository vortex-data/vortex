// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the cost of checking array-backed [`Validity`] per element.
//!
//! For the `Validity::Array` variant, `Validity::execute_is_valid` runs a scalar lookup through
//! the compute stack on *every* call, and the (now deprecated) `is_valid` additionally spins up a
//! fresh `ExecutionCtx` per call. Either way, calling it in a `for i in 0..n` loop is
//! pathologically slow. The fix used by callers is to materialize the validity into a `Mask` once
//! (`execute_mask`) and then do cheap O(1) bit reads via `Mask::value`. This bench contrasts the two.
//!
//! Sizes are kept small because the per-element variant is intentionally the slow one we are
//! measuring.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[256, 1024];

/// Build an array-backed validity (~10% nulls so it is `Validity::Array`).
fn array_validity(len: usize) -> Validity {
    Validity::from(BitBuffer::from_iter(
        (0..len).map(|i| !i.is_multiple_of(10)),
    ))
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// Per-element validity check over array-backed validity (the antipattern). This mirrors the
/// deprecated `Validity::is_valid(i)`: a fresh `ExecutionCtx` plus a scalar lookup on every call.
#[divan::bench(args = SIZES)]
fn is_valid_per_element(bencher: Bencher, len: usize) {
    let validity = array_validity(len);
    bencher
        .with_inputs(|| (&validity, SESSION.create_execution_ctx()))
        .bench_refs(|(validity, ctx)| {
            let mut count = 0usize;
            for i in 0..len {
                count += validity.execute_is_valid(i, ctx).unwrap() as usize;
            }
            count
        });
}

/// Materialize the validity into a `Mask` once, then read bits (the fix).
#[divan::bench(args = SIZES)]
fn execute_mask_then_value(bencher: Bencher, len: usize) {
    let validity = array_validity(len);
    bencher
        .with_inputs(|| (&validity, SESSION.create_execution_ctx()))
        .bench_refs(|(validity, ctx)| {
            let mask = validity.execute_mask(len, ctx).unwrap();
            let mut count = 0usize;
            for i in 0..len {
                count += mask.value(i) as usize;
            }
            count
        });
}
