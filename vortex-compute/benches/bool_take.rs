// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing optimized vs default bool take.
//!
//! The optimized take has special fast paths for:
//! - All true values → broadcast true
//! - All false values → broadcast false
//! - Single true value → comparison against that index
//! - Single false value → comparison and negate
//! - All null values → broadcast null
//! - Multiple true/false → fallback to default

use std::fmt;
use std::sync::LazyLock;

use itertools::Itertools;
use vortex_compute::take::Take;
use vortex_compute::take::default_take;
use vortex_compute::take::optimized_take;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::bool::BoolVectorMut;

fn main() {
    divan::main();
}

/// Value array patterns that exercise different optimization paths.
#[derive(Clone, Copy, Debug)]
enum ValuePattern {
    /// All values are true → optimized broadcasts true.
    AllTrue,
    /// All values are false → optimized broadcasts false.
    AllFalse,
    /// Single true among falses: [true, false, false, false] → optimized uses comparison.
    SingleTrue,
    /// Single false among trues: [false, true, true, true] → optimized uses comparison + negate.
    SingleFalse,
    /// Multiple true and false: [true, false, true, false] → falls back to default.
    Mixed,
    /// All values are null → optimized broadcasts null.
    AllNull,
    /// Single null with true: [null, true] → optimized broadcasts true (null skipped).
    SingleNullWithTrue,
    /// Single null with false: [null, false] → optimized broadcasts false (null skipped).
    SingleNullWithFalse,
    /// Mixed values with some nulls.
    MixedWithNulls,
}

impl ValuePattern {
    const ALL: &[Self] = &[
        Self::AllTrue,
        Self::AllFalse,
        Self::SingleTrue,
        Self::SingleFalse,
        Self::Mixed,
        Self::AllNull,
        Self::SingleNullWithTrue,
        Self::SingleNullWithFalse,
        Self::MixedWithNulls,
    ];

    fn create_values(self) -> BoolVector {
        match self {
            Self::AllTrue => BoolVectorMut::from_iter([Some(true), Some(true)]).freeze(),
            Self::AllFalse => BoolVectorMut::from_iter([Some(false), Some(false)]).freeze(),
            Self::SingleTrue => {
                // One true among multiple falses.
                BoolVectorMut::from_iter([Some(true), Some(false), Some(false), Some(false)])
                    .freeze()
            }
            Self::SingleFalse => {
                // One false among multiple trues.
                BoolVectorMut::from_iter([Some(false), Some(true), Some(true), Some(true)]).freeze()
            }
            Self::Mixed => {
                BoolVectorMut::from_iter([Some(true), Some(false), Some(true), Some(false)])
                    .freeze()
            }
            Self::AllNull => BoolVectorMut::from_iter([None, None]).freeze(),
            Self::SingleNullWithTrue => BoolVectorMut::from_iter([None, Some(true)]).freeze(),
            Self::SingleNullWithFalse => BoolVectorMut::from_iter([None, Some(false)]).freeze(),
            Self::MixedWithNulls => {
                BoolVectorMut::from_iter([Some(true), None, Some(false), None]).freeze()
            }
        }
    }

    fn max_index(self) -> usize {
        match self {
            Self::SingleTrue | Self::SingleFalse | Self::Mixed | Self::MixedWithNulls => 4,
            _ => 2,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::AllTrue => "all_true",
            Self::AllFalse => "all_false",
            Self::SingleTrue => "single_true",
            Self::SingleFalse => "single_false",
            Self::Mixed => "mixed",
            Self::AllNull => "all_null",
            Self::SingleNullWithTrue => "null_with_true",
            Self::SingleNullWithFalse => "null_with_false",
            Self::MixedWithNulls => "mixed_nulls",
        }
    }
}

const INDICES_SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// Benchmark parameters wrapper for Display impl.
#[derive(Clone, Copy)]
struct Params {
    indices_len: usize,
    pattern: ValuePattern,
}

impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.indices_len, self.pattern.name())
    }
}

static PARAMS: LazyLock<Vec<Params>> = LazyLock::new(|| {
    INDICES_SIZES
        .iter()
        .cartesian_product(ValuePattern::ALL.iter())
        .map(|(&indices_len, &pattern)| Params {
            indices_len,
            pattern,
        })
        .collect()
});

/// Creates indices that cycle through valid values for the pattern.
fn create_indices(len: usize, max_index: usize) -> Vec<u32> {
    #[expect(clippy::cast_possible_truncation)]
    (0..len).map(|i| (i % max_index) as u32).collect()
}

#[divan::bench(args = &*PARAMS, sample_count = 1000)]
fn default(bencher: divan::Bencher, params: &Params) {
    bencher
        .with_inputs(|| {
            let values = params.pattern.create_values();
            let indices = create_indices(params.indices_len, params.pattern.max_index());
            (values, indices)
        })
        .bench_refs(|(values, indices)| default_take(values, indices.as_slice()));
}

#[divan::bench(args = &*PARAMS, sample_count = 1000)]
fn optimized(bencher: divan::Bencher, params: &Params) {
    bencher
        .with_inputs(|| {
            let values = params.pattern.create_values();
            let indices = create_indices(params.indices_len, params.pattern.max_index());
            (values, indices)
        })
        .bench_refs(|(values, indices)| {
            optimized_take(values, indices.as_slice(), || {
                values.validity().take(indices.as_slice())
            })
        });
}
