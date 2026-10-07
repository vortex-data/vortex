// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares orderings for filtering a boolean dictionary with run-end codes.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]
#![expect(clippy::cast_sign_loss)]
#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    for &args in ARGS {
        let i = inputs(args);
        let mut ctx = SESSION.create_execution_ctx();
        let expected = filter_then_lookup_impl(&i);
        assert_arrays_eq!(lookup_materialize_filter_impl(&i), expected, &mut ctx);
        assert_arrays_eq!(lookup_runend_filter_impl(&i), expected, &mut ctx);
    }
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

const LEN: usize = 65_536;
const DICT_SIZE: usize = 64;

#[derive(Clone, Copy)]
enum MaskPattern {
    Random,
    Clustered,
}

#[derive(Clone, Copy)]
struct Args {
    /// Mean run length of the codes.
    run_length: usize,
    /// Fraction of rows selected by the filter.
    density: f64,
    pattern: MaskPattern,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pattern = match self.pattern {
            MaskPattern::Random => "rand",
            MaskPattern::Clustered => "clust",
        };
        write!(f, "run{}_d{}_{pattern}", self.run_length, self.density)
    }
}

const fn args(run_length: usize, density: f64, pattern: MaskPattern) -> Args {
    Args {
        run_length,
        density,
        pattern,
    }
}

const ARGS: &[Args] = &[
    args(4, 0.0001, MaskPattern::Random),
    args(4, 0.01, MaskPattern::Random),
    args(4, 0.1, MaskPattern::Random),
    args(4, 0.5, MaskPattern::Random),
    args(4, 0.9, MaskPattern::Random),
    args(16, 0.01, MaskPattern::Random),
    args(16, 0.1, MaskPattern::Random),
    args(16, 0.5, MaskPattern::Random),
    args(16, 0.9, MaskPattern::Random),
    args(64, 0.0001, MaskPattern::Random),
    args(64, 0.01, MaskPattern::Random),
    args(64, 0.1, MaskPattern::Random),
    args(64, 0.5, MaskPattern::Random),
    args(64, 0.9, MaskPattern::Random),
    args(1024, 0.0001, MaskPattern::Random),
    args(1024, 0.01, MaskPattern::Random),
    args(1024, 0.1, MaskPattern::Random),
    args(1024, 0.5, MaskPattern::Random),
    args(1024, 0.9, MaskPattern::Random),
    args(64, 0.01, MaskPattern::Clustered),
    args(64, 0.1, MaskPattern::Clustered),
    args(64, 0.5, MaskPattern::Clustered),
    args(1024, 0.1, MaskPattern::Clustered),
    args(1024, 0.5, MaskPattern::Clustered),
];

struct Inputs {
    codes: RunEndArray,
    values: ArrayRef,
    mask: Mask,
}

fn inputs(args: Args) -> Inputs {
    let mut rng = StdRng::seed_from_u64(0x5eed);

    // Run lengths are uniform in [1, 2 * run_length - 1] so boundaries are not word-aligned.
    let mut ends = Vec::new();
    let mut end = 0usize;
    while end < LEN {
        end = (end + rng.random_range(1..2 * args.run_length)).min(LEN);
        ends.push(end as u32);
    }
    let run_codes: PrimitiveArray = (0..ends.len())
        .map(|_| rng.random_range(0..DICT_SIZE as u32))
        .collect();
    let ends: PrimitiveArray = ends.into_iter().collect();
    let mut ctx = SESSION.create_execution_ctx();
    let codes = RunEnd::new(ends.into_array(), run_codes.into_array(), &mut ctx);
    let values = BoolArray::from_iter((0..DICT_SIZE).map(|i| i % 3 == 0)).into_array();

    let n_true = ((LEN as f64 * args.density).round() as usize).max(1);
    let bits: Vec<bool> = match args.pattern {
        MaskPattern::Random => {
            let mut bits = vec![false; LEN];
            bits.iter_mut().take(n_true).for_each(|b| *b = true);
            bits.shuffle(&mut rng);
            bits
        }
        MaskPattern::Clustered => {
            // Evenly spaced blocks of up to 256 selected rows.
            let block = 256.min(n_true);
            let stride = LEN / n_true.div_ceil(block);
            (0..LEN).map(|i| i % stride < block).collect()
        }
    };

    Inputs {
        codes,
        values,
        mask: Mask::from(BitBuffer::from(bits)),
    }
}

/// Current order: filter integer run-end codes, then look up the dictionary.
fn filter_then_lookup_impl(i: &Inputs) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let codes = i.codes.clone().into_array().filter(i.mask.clone()).unwrap();
    DictArray::new(codes, i.values.clone())
        .into_array()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

/// Look up every run, materialize booleans, then filter.
fn lookup_materialize_filter_impl(i: &Inputs) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    DictArray::new(i.codes.clone().into_array(), i.values.clone())
        .into_array()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
        .filter(i.mask.clone())
        .unwrap()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

/// Look up each run into a run-end boolean array, then filter it lazily.
fn lookup_runend_filter_impl(i: &Inputs) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let bool_values = i.values.take(i.codes.values().clone()).unwrap();
    // SAFETY: ends are unchanged from a valid RunEndArray and values have one entry per run.
    let ree = unsafe {
        RunEnd::new_unchecked(
            i.codes.ends().clone(),
            bool_values,
            i.codes.offset(),
            i.codes.len(),
        )
    };
    ree.into_array()
        .filter(i.mask.clone())
        .unwrap()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

#[divan::bench(args = ARGS)]
fn filter_then_lookup(bencher: Bencher, args: Args) {
    let i = inputs(args);
    bencher.bench(|| filter_then_lookup_impl(&i));
}

#[divan::bench(args = ARGS)]
fn lookup_materialize_filter(bencher: Bencher, args: Args) {
    let i = inputs(args);
    bencher.bench(|| lookup_materialize_filter_impl(&i));
}

#[divan::bench(args = ARGS)]
fn lookup_runend_filter(bencher: Bencher, args: Args) {
    let i = inputs(args);
    bencher.bench(|| lookup_runend_filter_impl(&i));
}
