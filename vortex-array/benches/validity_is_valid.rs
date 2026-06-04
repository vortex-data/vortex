// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the cost of reading array-backed [`Validity`] per element.
//!
//! `Validity::is_valid(i)` for the `Validity::Array` variant spins up a fresh
//! execution context and executes a scalar lookup on *every* call, so calling it
//! in a `for i in 0..n` loop is pathologically slow. The fix used by callers is to
//! materialize the validity into a `Mask` once (`execute_mask`) and then do cheap
//! O(1) bit reads via `Mask::value`. This bench contrasts the two.
//!
//! Sizes are kept small because the per-element variant is intentionally the slow
//! one we are measuring.

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[256, 1024, 4096];

/// Build an array-backed validity (~10% nulls so it is `Validity::Array`).
fn array_validity(len: usize) -> Validity {
    Validity::from(BitBuffer::from_iter(
        (0..len).map(|i| !i.is_multiple_of(10)),
    ))
}

/// Per-element `Validity::is_valid` over array-backed validity (the antipattern).
#[divan::bench(args = SIZES)]
fn is_valid_per_element(bencher: Bencher, len: usize) {
    let validity = array_validity(len);
    bencher.with_inputs(|| &validity).bench_refs(|validity| {
        let mut count = 0usize;
        for i in 0..len {
            count += validity.is_valid(i).unwrap() as usize;
        }
        count
    });
}

/// Materialize the validity into a `Mask` once, then read bits (the fix).
#[divan::bench(args = SIZES)]
fn execute_mask_then_value(bencher: Bencher, len: usize) {
    let validity = array_validity(len);
    bencher.with_inputs(|| &validity).bench_refs(|validity| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mask = validity.execute_mask(len, &mut ctx).unwrap();
        let mut count = 0usize;
        for i in 0..len {
            count += mask.value(i) as usize;
        }
        count
    });
}
