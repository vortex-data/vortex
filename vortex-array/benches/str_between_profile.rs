// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Profile bench: where does str_between_sorted spend its time?
//!
//! Splits the sorted-dict `BETWEEN` path into its stages and times each one alone:
//!   - scan_sorted_bounds for one needle on the dict values (varbinview, 1024 strings)
//!   - primitive Between on the codes (u16[N]) given pre-computed bounds
//!   - the full plain dict path
//!   - the full sorted dict path
//!
//! Lets us see whether the bottleneck is the value scan, the codes-domain Between
//! kernel, or the executor scaffolding.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builders::dict::sort_dict;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::between;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[100_000];
const UNIQUES: usize = 1024;

fn make_sorted_dict(len: usize) -> DictArray {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, UNIQUES));
    let dict = dict_encode(&arr.into_array()).unwrap();
    sort_dict(dict).unwrap()
}

// ---------------------------------------------------------------------------
// Component 1: scan_sorted_bounds on the values for a single needle.
// We can't call the private helper directly, but we can call binary(values, scalar, Lt)
// which forces the scan through the kernel path (same scan code).
// ---------------------------------------------------------------------------
#[divan::bench(args = SIZES)]
fn just_values_lt_scan(bencher: Bencher, len: usize) {
    use vortex_array::arrays::dict::DictArraySlotsExt;
    let dict = make_sorted_dict(len);
    let values = dict.values().clone();
    let n = values.len();
    bencher
        .with_inputs(|| (values.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(v, mut ctx)| {
            v.binary(ConstantArray::new("k", n).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// Component 2: raw primitive Between on a u16 codes array, no dict involved.
// Times the BetweenKernel for Primitive directly.
// ---------------------------------------------------------------------------
#[divan::bench(args = SIZES)]
fn primitive_between_codes(bencher: Bencher, len: usize) {
    let codes: PrimitiveArray = (0..len).map(|i| (i % UNIQUES) as u16).collect();
    let codes_arr = codes.into_array();
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    bencher
        .with_inputs(|| (codes_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(c, mut ctx)| {
            let lo = ConstantArray::new(256u16, len).into_array();
            let hi = ConstantArray::new(768u16, len).into_array();
            Between
                .try_new_array(len, opts.clone(), [c, lo, hi])
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// Component 3: full sorted dict between via the apply(between) expression.
// ---------------------------------------------------------------------------
#[divan::bench(args = SIZES)]
fn full_sorted_dict_between(bencher: Bencher, len: usize) {
    let dict = make_sorted_dict(len).into_array();
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit("k"), lit("p"), opts);
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap());
}

// ---------------------------------------------------------------------------
// Component 4: plain dict between (no sort).
// ---------------------------------------------------------------------------
#[divan::bench(args = SIZES)]
fn full_plain_dict_between(bencher: Bencher, len: usize) {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(len, UNIQUES));
    let dict = dict_encode(&arr.into_array()).unwrap().into_array();
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit("k"), lit("p"), opts);
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap());
}

// ---------------------------------------------------------------------------
// Component 5: full sorted dict Lt for comparison.
// ---------------------------------------------------------------------------
#[divan::bench(args = SIZES)]
fn full_sorted_dict_lt(bencher: Bencher, len: usize) {
    let dict = make_sorted_dict(len).into_array();
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new("m", len).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

// Silence unused warnings.
const _: fn() = || {
    let _ = std::marker::PhantomData::<Primitive>;
};
