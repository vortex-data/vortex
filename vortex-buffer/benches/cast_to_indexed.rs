// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Focused bench for the **best fallible cast kernel** — what `cast.rs` actually uses
//! in `vortex-array/src/arrays/primitive/compute/cast.rs`. Single bench, no cross-impl
//! baselines: just a regression guard for the production cast hot path.
//!
//! The kernel: [`vortex_buffer::lane_ops_indexed::try_map_with_mask`] called with a
//! lazy-validity `or_else` closure — for statically-infallible casts (widening) LLVM
//! proves `NumCast::from` is always `Some`, the `or_else` branch is dead, and the
//! validity path is DCE'd. For fallible casts (narrowing), validity is only consulted
//! on the cold failure branch.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use divan::Bencher;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::prelude::*;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576];
const VALID_RATE: f64 = 0.7;
const DATA_SEED: u64 = 0;
const VALID_SEED: u64 = 1;

struct Fixture {
    values: Buffer<u64>,
    mask: BitBuffer,
}

fn fixture(n: usize) -> Fixture {
    let mut data_rng = StdRng::seed_from_u64(DATA_SEED);
    let mut valid_rng = StdRng::seed_from_u64(VALID_SEED);
    let raw_values: Vec<u64> = (0..n)
        .map(|_| data_rng.random_range(0..u32::MAX as u64))
        .collect();
    let raw_valid: Vec<bool> = (0..n).map(|_| valid_rng.random_bool(VALID_RATE)).collect();

    let values: Buffer<u64> = raw_values.iter().copied().collect();
    let mask = {
        let mut m = BitBufferMut::with_capacity(n);
        for &v in &raw_valid {
            m.append(v);
        }
        m.freeze()
    };

    Fixture { values, mask }
}

/// The kernel `cast.rs` uses in production: `try_map_with_mask` with a lazy-validity
/// `or_else` closure. `NumCast::from(v)` is the cast; `or_else` only fires (and only
/// then reads `valid`) when the cast itself returned `None`.
#[divan::bench(args = SIZES)]
fn cast_lazy_validity(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            let mut out: Vec<MaybeUninit<u32>> = Vec::with_capacity(n);
            // SAFETY: every lane is written before any read inside the kernel.
            unsafe { out.set_len(n) };
            (f.values.clone(), f.mask.clone(), out)
        })
        .bench_refs(|(values, mask, out)| {
            try_map_with_mask(values.as_slice(), mask, out.as_mut_slice(), |v, valid| {
                <u32 as NumCast>::from(v).or_else(|| (!valid).then(u32::default))
            })
            .unwrap();
        });
}
