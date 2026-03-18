// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Micro-benchmarks to measure dict decode overhead at each layer:
//!
//! - `take_bc_{type}` — take_canonical with pre-canonicalized inputs.
//!    Measures: Canonical match + dict_take_validity + kernel. No execute loop.
//! - `take_old_{type}` — old TakeExecute dispatch with pre-canonicalized inputs.
//!    Measures: codes.clone().into_array() + TakeExecute trait dispatch + kernel.
//! - `full_decode_{type}` — dict.to_canonical() end-to-end.
//!    Measures: execute loop + canonicalization + kernel.
//!
//! B savings (direct kernel) = take_old - take_bc
//! A savings (execute loop shortcut) = full_decode - take_bc

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeExecute;
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
];

// ── Primitive benchmarks ────────────────────────────────────────────────────

/// Current optimized take_canonical (direct kernel + validity shortcut).
#[divan::bench(types = [f32, i64], args = BENCH_ARGS)]
fn take_bc_primitive<T>(bencher: Bencher, (len, unique_values): (usize, usize))
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

/// Old TakeExecute dispatch path (pre-optimization baseline).
#[divan::bench(types = [f32, i64], args = BENCH_ARGS)]
fn take_old_primitive<T>(bencher: Bencher, (len, unique_values): (usize, usize))
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let arr = gen_primitive_for_dict::<T>(len, unique_values);
    let dict = dict_encode(&arr.into_array()).unwrap();
    let values = dict.values().to_primitive();
    let codes = dict.codes().to_primitive();

    bencher
        .with_inputs(|| (&values, &codes))
        .bench_refs(|(values, codes)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            <Primitive as TakeExecute>::take(values, &codes.clone().into_array(), &mut ctx)
        });
}

/// Full dict decode via to_canonical() — execute loop + canonicalization + kernel.
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

/// Current optimized take_canonical for VarBinView.
#[divan::bench(args = BENCH_ARGS)]
fn take_bc_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
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

/// Old TakeExecute dispatch path for VarBinView.
#[divan::bench(args = BENCH_ARGS)]
fn take_old_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&arr.into_array()).unwrap();
    let values = dict.values().to_varbinview();
    let codes = dict.codes().to_primitive();

    bencher
        .with_inputs(|| (&values, &codes))
        .bench_refs(|(values, codes)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            <VarBinView as TakeExecute>::take(values, &codes.clone().into_array(), &mut ctx)
        });
}

/// Full dict decode for VarBinView.
#[divan::bench(args = BENCH_ARGS)]
fn full_decode_varbinview(bencher: Bencher, (len, unique_values): (usize, usize)) {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, unique_values));
    let dict = dict_encode(&arr.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}
