// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Micro-benchmarks to isolate dict decode overhead at different levels:
//!
//! 1. `take_canonical_only` — take_canonical with pre-canonicalized values+codes
//!    (dict type dispatch + validity + kernel, NO execute loop)
//! 2. `full_decode` — dict.to_canonical() including execute loop + codes/values
//!    canonicalization
//!
//! The delta between full_decode and take_canonical_only measures the execute
//! loop overhead (optimize, matcher checks, codes/values canonicalization).

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::take_canonical;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::builders::dict::dict_encode;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::NativePType;

fn main() {
    warm_up_vtables();
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    (1_000, 4),
    (1_000, 32),
    (10_000, 4),
    (10_000, 32),
    (10_000, 512),
    (100_000, 32),
    (100_000, 512),
];

// ── Primitive benchmarks ────────────────────────────────────────────────────

/// take_canonical with pre-canonicalized values+codes.
/// Measures: Canonical enum match + dict_take_validity + primitive gather kernel.
/// Does NOT include execute loop, codes/values canonicalization, or optimize().
#[divan::bench(types = [f32, i64], args = BENCH_ARGS)]
fn take_canonical_primitive<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(&arr.into_array()).unwrap();
    let values = Canonical::Primitive(dict.values().to_primitive());
    let codes = dict.codes().to_primitive();

    bencher
        .with_inputs(|| (values.clone(), &codes))
        .bench_refs(|(values, codes)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            take_canonical(values.clone(), codes, &mut ctx)
        });
}

/// Full dict decode via to_canonical() — execute loop + canonicalization + kernel.
/// The delta from take_canonical_primitive measures execute loop overhead.
#[divan::bench(types = [f32, i64], args = BENCH_ARGS)]
fn full_decode_primitive<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(&arr.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}

// ── VarBinView benchmarks ───────────────────────────────────────────────────

/// take_canonical for VarBinView with pre-canonicalized inputs.
/// Measures: enum match + dict_take_validity + codes_to_mask + view gather.
#[divan::bench(args = BENCH_ARGS)]
fn take_canonical_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&arr.into_array()).unwrap();
    let values = Canonical::VarBinView(dict.values().to_varbinview());
    let codes = dict.codes().to_primitive();

    bencher
        .with_inputs(|| (values.clone(), &codes))
        .bench_refs(|(values, codes)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            take_canonical(values.clone(), codes, &mut ctx)
        });
}

/// Full dict decode for VarBinView.
/// The delta from take_canonical_varbinview measures execute loop overhead.
#[divan::bench(args = BENCH_ARGS)]
fn full_decode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&arr.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}
