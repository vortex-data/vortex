// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the Vortex [`Interleave`](vortex_array::arrays::Interleave) boolean execute path
//! against the equivalent arrow-rs [`interleave`] kernel, across several access patterns:
//!
//! - `random`: fully random `(array_index, row_index)` per output row.
//! - `round_robin`: a merge — `array_index = i % N`, `row_index = i / N`, each branch consumed
//!   front-to-back; the structured pattern most amenable to both engines.
//! - `single_branch`: every row routed to branch 0 with a random row (a degenerate gather).
//!
//! Each pattern is run for `N = 2` and `N = 4`, nullable and non-nullable, against identical data
//! so the two engines are directly comparable.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray;
use arrow_select::interleave::interleave;
use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::{array_session, ArrayRef};
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::InterleaveArray;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 8_192;

/// A single benchmark configuration: access pattern, branch count, and nullability.
#[derive(Clone, Copy)]
struct Combo {
    pattern: &'static str,
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

/// The full cross product of patterns × branch counts × nullability that every engine is measured
/// against, so the Vortex and arrow rows line up one-to-one in the report.
fn combos() -> Vec<Combo> {
    let mut out = Vec::new();
    for pattern in ["random", "round_robin", "single_branch"] {
        for branches in [2, 4] {
            for nullable in [false, true] {
                out.push(Combo {
                    pattern,
                    branches,
                    nullable,
                });
            }
        }
    }
    out
}

/// Per-branch boolean values (as `Option<bool>`, all `Some` when non-nullable) paired with the
/// `(array_index, row_index)` selectors for one [`Combo`].
type Inputs = (Vec<Vec<Option<bool>>>, Vec<(usize, usize)>);

/// Generates the shared [`Inputs`] for a [`Combo`].
///
/// Seeded only by the combo so both engines see byte-identical data for a fair comparison.
fn gen_inputs(combo: Combo) -> Inputs {
    let mut rng = StdRng::seed_from_u64(0);
    let bit = Uniform::new(0u8, 2).unwrap();

    let values: Vec<Vec<Option<bool>>> = (0..combo.branches)
        .map(|_| {
            (0..ARRAY_SIZE)
                .map(|_| {
                    if combo.nullable && rng.sample(bit) == 0 {
                        None
                    } else {
                        Some(rng.sample(bit) == 0)
                    }
                })
                .collect()
        })
        .collect();

    let branch = Uniform::new(0usize, combo.branches).unwrap();
    let row = Uniform::new(0usize, ARRAY_SIZE).unwrap();
    let selectors: Vec<(usize, usize)> = (0..ARRAY_SIZE)
        .map(|i| match combo.pattern {
            "random" => (rng.sample(branch), rng.sample(row)),
            "round_robin" => (i % combo.branches, (i / combo.branches) % ARRAY_SIZE),
            "single_branch" => (0, rng.sample(row)),
            other => unreachable!("unknown pattern {other}"),
        })
        .collect();

    (values, selectors)
}

/// Builds the Vortex value arrays and the `u32` selector buffers for a [`Combo`].
fn vortex_inputs(combo: Combo) -> (Vec<ArrayRef>, Buffer<u32>, Buffer<u32>) {
    let (values, selectors) = gen_inputs(combo);
    let values = values
        .into_iter()
        .map(|vals| {
            if combo.nullable {
                BoolArray::from_iter(vals).into_array()
            } else {
                BoolArray::from_iter(vals.into_iter().map(Option::unwrap)).into_array()
            }
        })
        .collect();
    let array_indices = selectors
        .iter()
        .map(|&(a, _)| u32::try_from(a).unwrap())
        .collect();
    let row_indices = selectors
        .iter()
        .map(|&(_, r)| u32::try_from(r).unwrap())
        .collect();
    (values, array_indices, row_indices)
}

/// Builds the arrow value arrays and the `(usize, usize)` selectors for a [`Combo`].
fn arrow_inputs(combo: Combo) -> (Vec<ArrowArrayRef>, Vec<(usize, usize)>) {
    let (values, selectors) = gen_inputs(combo);
    let values = values
        .into_iter()
        .map(|vals| -> ArrowArrayRef {
            if combo.nullable {
                Arc::new(BooleanArray::from(vals))
            } else {
                Arc::new(BooleanArray::from(
                    vals.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                ))
            }
        })
        .collect();
    (values, selectors)
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

#[divan::bench(args = combos())]
fn arrow(bencher: Bencher, combo: Combo) {
    let (values, selectors) = arrow_inputs(combo);
    bencher.bench(|| {
        let refs: Vec<&dyn ArrowArray> = values.iter().map(AsRef::as_ref).collect();
        interleave(&refs, &selectors).unwrap()
    });
}
