// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbench: compare ops and BETWEEN on sorted vs unsorted dict, across types and
//! shapes. Each pair `<op>_plain` / `<op>_sorted` runs the same predicate against an
//! unsorted dict and a sorted dict.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builders::dict::sort_dict;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::expr::between;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const ARGS: &[(usize, usize)] = &[
    // (codes_len, uniques). Uniques >= 256 forces u16 codes.
    (10_000, 256),
    (10_000, 1024),
    (100_000, 256),
    (100_000, 1024),
];

// Baselines that bypass the dict layer:
//   raw_u16_lt:              SIMD primitive cmp (what the sorted dict emits)
//   raw_take_bool:           take(bool[1024], codes_u16) (what plain dict emits)
//   dict_take_via_pushdown:  the full plain-dict push-down shape, end-to-end

const BASELINE_SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 5_000_000];

#[divan::bench(args = BASELINE_SIZES)]
fn raw_u16_lt_baseline(bencher: Bencher, len: usize) {
    use vortex_array::arrays::PrimitiveArray;
    #[allow(clippy::cast_possible_truncation, reason = "u16 dict size by construction")]
    let arr: PrimitiveArray = (0..len).map(|i| (i % 1024) as u16).collect();
    let arr = arr.into_array();
    bencher
        .with_inputs(|| (arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| {
            a.binary(ConstantArray::new(512u16, len).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = BASELINE_SIZES)]
fn raw_take_bool_baseline(bencher: Bencher, len: usize) {
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use rand::RngExt;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    let bools: BoolArray = (0..1024).map(|i| i % 7 == 0).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let codes: PrimitiveArray = (0..len).map(|_| rng.random_range(0u16..1024)).collect();
    let bools = bools.into_array();
    let codes = codes.into_array();
    bencher
        .with_inputs(|| {
            (
                bools.clone(),
                codes.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(b, c, mut ctx)| b.take(c).unwrap().execute::<Mask>(&mut ctx).unwrap());
}

// End-to-end shape the push-down rule produces:
//   DictArray<codes_u16[N], ScalarFnArray<Binary, [values_i64[1024], const]>>
#[divan::bench(args = BASELINE_SIZES)]
fn dict_take_via_pushdown_baseline(bencher: Bencher, len: usize) {
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use rand::SeedableRng;
    use rand::RngExt;
    use rand::prelude::StdRng;
    let mut rng = StdRng::seed_from_u64(0);
    let values: PrimitiveArray = (0..1024i64).map(|_| rng.random::<i64>()).collect();
    let codes: PrimitiveArray = (0..len).map(|_| rng.random_range(0u16..1024)).collect();
    let dict = DictArray::try_new(codes.into_array(), values.into_array())
        .unwrap()
        .into_array();
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new(0i64, len).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// Primitive (i64, u8, f32): compare with Eq, Lt, Gte; BETWEEN
// ---------------------------------------------------------------------------

fn run_primitive_compare<T>(
    bencher: Bencher,
    args: (usize, usize),
    sorted: bool,
    op: Operator,
    needle: T,
) where
    T: NativePType + Into<Scalar>,
    StandardUniform: Distribution<T>,
    Scalar: From<T>,
{
    let (len, uniques) = args;
    let arr = gen_primitive_for_dict::<T>(len, uniques).into_array();
    let dict: DictArray = dict_encode(&arr).unwrap();
    let dict_arr = if sorted {
        sort_dict(dict).unwrap().into_array()
    } else {
        dict.into_array()
    };
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new(needle, len).into_array(), op)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = ARGS)]
fn i64_eq_plain(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, false, Operator::Eq, 42i64);
}
#[divan::bench(args = ARGS)]
fn i64_eq_sorted(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, true, Operator::Eq, 42i64);
}

#[divan::bench(args = ARGS)]
fn i64_lt_plain(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, false, Operator::Lt, i64::MAX / 2);
}
#[divan::bench(args = ARGS)]
fn i64_lt_sorted(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, true, Operator::Lt, i64::MAX / 2);
}

#[divan::bench(args = ARGS)]
fn i64_gte_plain(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, false, Operator::Gte, i64::MAX / 2);
}
#[divan::bench(args = ARGS)]
fn i64_gte_sorted(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<i64>(bencher, args, true, Operator::Gte, i64::MAX / 2);
}

#[divan::bench(args = ARGS)]
fn u8_eq_plain(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<u8>(bencher, args, false, Operator::Eq, 42u8);
}
#[divan::bench(args = ARGS)]
fn u8_eq_sorted(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<u8>(bencher, args, true, Operator::Eq, 42u8);
}

#[divan::bench(args = ARGS)]
fn f32_lt_plain(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<f32>(bencher, args, false, Operator::Lt, 0.5f32);
}
#[divan::bench(args = ARGS)]
fn f32_lt_sorted(bencher: Bencher, args: (usize, usize)) {
    run_primitive_compare::<f32>(bencher, args, true, Operator::Lt, 0.5f32);
}

// ---------------------------------------------------------------------------
// String compare
// ---------------------------------------------------------------------------

fn run_str_compare(
    bencher: Bencher,
    args: (usize, usize),
    sorted: bool,
    op: Operator,
    needle: &'static str,
) {
    let (len, uniques) = args;
    let varbinview = VarBinViewArray::from_iter_str(gen_varbin_words(len, uniques));
    let dict: DictArray = dict_encode(&varbinview.into_array()).unwrap();
    let dict_arr = if sorted {
        sort_dict(dict).unwrap().into_array()
    } else {
        dict.into_array()
    };
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new(needle, len).into_array(), op)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = ARGS)]
fn str_eq_plain(bencher: Bencher, args: (usize, usize)) {
    run_str_compare(bencher, args, false, Operator::Eq, "x");
}
#[divan::bench(args = ARGS)]
fn str_eq_sorted(bencher: Bencher, args: (usize, usize)) {
    run_str_compare(bencher, args, true, Operator::Eq, "x");
}

#[divan::bench(args = ARGS)]
fn str_lt_plain(bencher: Bencher, args: (usize, usize)) {
    run_str_compare(bencher, args, false, Operator::Lt, "m");
}
#[divan::bench(args = ARGS)]
fn str_lt_sorted(bencher: Bencher, args: (usize, usize)) {
    run_str_compare(bencher, args, true, Operator::Lt, "m");
}

// Also benchmark VarBin (non-view) for parity with existing benches.
#[divan::bench(args = ARGS)]
fn varbin_lt_plain(bencher: Bencher, args: (usize, usize)) {
    let (len, uniques) = args;
    let arr = VarBinArray::from(gen_varbin_words(len, uniques));
    let dict: DictArray = dict_encode(&arr.into_array()).unwrap();
    let dict_arr = dict.into_array();
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new("m", len).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}
#[divan::bench(args = ARGS)]
fn varbin_lt_sorted(bencher: Bencher, args: (usize, usize)) {
    let (len, uniques) = args;
    let arr = VarBinArray::from(gen_varbin_words(len, uniques));
    let dict: DictArray = dict_encode(&arr.into_array()).unwrap();
    let dict_arr = sort_dict(dict).unwrap().into_array();
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.binary(ConstantArray::new("m", len).into_array(), Operator::Lt)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// BETWEEN
// ---------------------------------------------------------------------------

fn run_between_primitive<T>(
    bencher: Bencher,
    args: (usize, usize),
    sorted: bool,
    lo: T,
    hi: T,
) where
    T: NativePType + Into<Scalar> + Copy,
    StandardUniform: Distribution<T>,
    Scalar: From<T>,
{
    let (len, uniques) = args;
    let arr = gen_primitive_for_dict::<T>(len, uniques).into_array();
    let dict: DictArray = dict_encode(&arr).unwrap();
    let dict_arr = if sorted {
        sort_dict(dict).unwrap().into_array()
    } else {
        dict.into_array()
    };
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit(lo), lit(hi), opts);
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = ARGS)]
fn i64_between_plain(bencher: Bencher, args: (usize, usize)) {
    run_between_primitive::<i64>(bencher, args, false, -i64::MAX / 4, i64::MAX / 4);
}
#[divan::bench(args = ARGS)]
fn i64_between_sorted(bencher: Bencher, args: (usize, usize)) {
    run_between_primitive::<i64>(bencher, args, true, -i64::MAX / 4, i64::MAX / 4);
}

#[divan::bench(args = ARGS)]
fn f32_between_plain(bencher: Bencher, args: (usize, usize)) {
    run_between_primitive::<f32>(bencher, args, false, 0.25f32, 0.75f32);
}
#[divan::bench(args = ARGS)]
fn f32_between_sorted(bencher: Bencher, args: (usize, usize)) {
    run_between_primitive::<f32>(bencher, args, true, 0.25f32, 0.75f32);
}

#[divan::bench(args = ARGS)]
fn str_between_plain(bencher: Bencher, args: (usize, usize)) {
    let (len, uniques) = args;
    let varbinview = VarBinViewArray::from_iter_str(gen_varbin_words(len, uniques));
    let dict: DictArray = dict_encode(&varbinview.into_array()).unwrap();
    let dict_arr = dict.into_array();
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit("k"), lit("p"), opts);
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap()
        });
}
#[divan::bench(args = ARGS)]
fn str_between_sorted(bencher: Bencher, args: (usize, usize)) {
    let (len, uniques) = args;
    let varbinview = VarBinViewArray::from_iter_str(gen_varbin_words(len, uniques));
    let dict: DictArray = dict_encode(&varbinview.into_array()).unwrap();
    let dict_arr = sort_dict(dict).unwrap().into_array();
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit("k"), lit("p"), opts);
    bencher
        .with_inputs(|| (dict_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| {
            d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap()
        });
}

// Silence the bencher's unused warnings for varbin (we use it to drive a single bench).
const _: fn() = || {
    let _ = size_of::<VarBinArray>();
};
