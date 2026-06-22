// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the Vortex [`Interleave`](vortex_array::arrays::Interleave) boolean execute path on a
//! focused set of configurations:
//!
//! - `round_robin`, 2 children: a merge — `array_index = i % N`, `row_index = i / N`.
//! - `random`, 2 children: fully random `(array_index, row_index)` per output row.
//! - `random`, 64 children: the same random gather spread over many value arrays.
//!
//! Each is run nullable and non-nullable.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::InterleaveArray;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 8_192;

/// The access pattern used to generate the `(array_index, row_index)` selectors.
#[derive(Clone, Copy)]
enum Pattern {
    /// A merge: `array_index = i % N`, `row_index = i / N`.
    RoundRobin,
    /// Fully random `(array_index, row_index)` per output row.
    Random,
}

impl Display for Pattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Pattern::RoundRobin => "round_robin",
            Pattern::Random => "random",
        })
    }
}

/// A single benchmark configuration: access pattern, branch count, and nullability.
#[derive(Clone, Copy)]
struct Combo {
    pattern: Pattern,
    branches: usize,
    nullable: bool,
}

impl Display for Combo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/n{}/{}",
            self.pattern,
            self.branches,
            if self.nullable { "null" } else { "nonnull" }
        )
    }
}

/// The configurations the benchmark covers: 2-child round-robin, 2-child random, and 64-child
/// random — each nullable and non-nullable.
fn combos() -> Vec<Combo> {
    let mut out = Vec::new();
    for nullable in [false, true] {
        for (pattern, branches) in [
            (Pattern::RoundRobin, 2),
            (Pattern::Random, 2),
            (Pattern::Random, 64),
        ] {
            out.push(Combo {
                pattern,
                branches,
                nullable,
            });
        }
    }
    out
}

/// Builds the Vortex value arrays and the `u32` selector buffers for a [`Combo`].
///
/// Seeded only by the combo so a run is deterministic and comparable across revisions.
fn vortex_inputs(combo: Combo) -> (Vec<ArrayRef>, Buffer<u32>, Buffer<u32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let bit = Uniform::new(0u8, 2).unwrap();

    let values = (0..combo.branches)
        .map(|_| {
            if combo.nullable {
                BoolArray::from_iter(
                    (0..ARRAY_SIZE).map(|_| (rng.sample(bit) == 0).then_some(rng.sample(bit) == 0)),
                )
                .into_array()
            } else {
                BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(bit) == 0)).into_array()
            }
        })
        .collect();

    let branch = Uniform::new(0u32, u32::try_from(combo.branches).unwrap()).unwrap();
    let row = Uniform::new(0u32, u32::try_from(ARRAY_SIZE).unwrap()).unwrap();
    let array_indices: Buffer<u32> = (0..ARRAY_SIZE)
        .map(|i| match combo.pattern {
            Pattern::Random => rng.sample(branch),
            Pattern::RoundRobin => u32::try_from(i % combo.branches).unwrap(),
        })
        .collect();
    let row_indices: Buffer<u32> = (0..ARRAY_SIZE)
        .map(|i| match combo.pattern {
            Pattern::Random => rng.sample(row),
            Pattern::RoundRobin => u32::try_from((i / combo.branches) % ARRAY_SIZE).unwrap(),
        })
        .collect();
    (values, array_indices, row_indices)
}

#[divan::bench(args = combos())]
fn vortex(bencher: Bencher, combo: Combo) {
    let (values, array_indices, row_indices) = vortex_inputs(combo);
    let session = array_session();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx));
}
