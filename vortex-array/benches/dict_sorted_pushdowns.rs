// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comprehensive bench of every pushdown a sorted dict can accelerate.
//!
//! Each pair `<op>_plain` / `<op>_sorted` runs the same predicate against an unsorted
//! dict and a sorted dict; the table at the end shows the relative cost.
//!
//! Pushdowns covered:
//!   - compare (eq, neq, lt, lte, gt, gte) with a scalar
//!   - between
//!   - LIKE 'prefix%'
//!   - min_max
//!   - is_sorted

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builders::dict::sort_dict;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::between;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison::NonStrict;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const N: usize = 100_000;
const UNIQUES: usize = 1024;

fn primitive_dict<T>(sorted: bool) -> vortex_array::ArrayRef
where
    T: vortex_array::dtype::NativePType,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    let arr = gen_primitive_for_dict::<T>(N, UNIQUES).into_array();
    let dict = dict_encode(&arr).unwrap();
    if sorted {
        sort_dict(dict).unwrap().into_array()
    } else {
        dict.into_array()
    }
}

fn str_dict(sorted: bool) -> vortex_array::ArrayRef {
    let arr = VarBinViewArray::from_iter_str(gen_varbin_words(N, UNIQUES));
    let dict = dict_encode(&arr.into_array()).unwrap();
    if sorted {
        sort_dict(dict).unwrap().into_array()
    } else {
        dict.into_array()
    }
}

// ---------------------------------------------------------------------------
// Compare ops
// ---------------------------------------------------------------------------

macro_rules! compare_bench {
    ($name:ident, $T:ty, $needle:expr, $op:expr, $sorted:expr) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            let dict = primitive_dict::<$T>($sorted);
            let scalar = ConstantArray::new($needle, N).into_array();
            bencher
                .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
                .bench_values(|(d, mut ctx)| {
                    d.binary(scalar.clone(), $op)
                        .unwrap()
                        .execute::<Mask>(&mut ctx)
                        .unwrap()
                });
        }
    };
}

compare_bench!(i64_eq_plain, i64, 42i64, Operator::Eq, false);
compare_bench!(i64_eq_sorted, i64, 42i64, Operator::Eq, true);
compare_bench!(i64_neq_plain, i64, 42i64, Operator::NotEq, false);
compare_bench!(i64_neq_sorted, i64, 42i64, Operator::NotEq, true);
compare_bench!(i64_lt_plain, i64, i64::MAX / 2, Operator::Lt, false);
compare_bench!(i64_lt_sorted, i64, i64::MAX / 2, Operator::Lt, true);
compare_bench!(i64_lte_plain, i64, i64::MAX / 2, Operator::Lte, false);
compare_bench!(i64_lte_sorted, i64, i64::MAX / 2, Operator::Lte, true);
compare_bench!(i64_gt_plain, i64, i64::MAX / 2, Operator::Gt, false);
compare_bench!(i64_gt_sorted, i64, i64::MAX / 2, Operator::Gt, true);
compare_bench!(i64_gte_plain, i64, i64::MAX / 2, Operator::Gte, false);
compare_bench!(i64_gte_sorted, i64, i64::MAX / 2, Operator::Gte, true);

// ---------------------------------------------------------------------------
// Between
// ---------------------------------------------------------------------------

fn run_between_i64(bencher: Bencher, sorted: bool) {
    let dict = primitive_dict::<i64>(sorted);
    let opts = BetweenOptions {
        lower_strict: NonStrict,
        upper_strict: NonStrict,
    };
    let expr = between(root(), lit(-i64::MAX / 4), lit(i64::MAX / 4), opts);
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| d.apply(&expr).unwrap().execute::<Mask>(&mut ctx).unwrap());
}
#[divan::bench]
fn i64_between_plain(bencher: Bencher) { run_between_i64(bencher, false); }
#[divan::bench]
fn i64_between_sorted(bencher: Bencher) { run_between_i64(bencher, true); }

// ---------------------------------------------------------------------------
// LIKE 'prefix%'
// ---------------------------------------------------------------------------

fn run_like(bencher: Bencher, sorted: bool, pattern: &'static str) {
    let dict = str_dict(sorted);
    let pattern_arr = ConstantArray::new(pattern, N).into_array();
    bencher
        .with_inputs(|| (dict.clone(), pattern_arr.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, p, mut ctx)| {
            Like.try_new_array(N, LikeOptions::default(), [d, p])
                .unwrap()
                .optimize()
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
        });
}
#[divan::bench]
fn str_like_prefix_plain(bencher: Bencher) { run_like(bencher, false, "abc%"); }
#[divan::bench]
fn str_like_prefix_sorted(bencher: Bencher) { run_like(bencher, true, "abc%"); }
// Non-prefix patterns: should not benefit from sorted-aware dispatch.
#[divan::bench]
fn str_like_middle_plain(bencher: Bencher) { run_like(bencher, false, "%abc%"); }
#[divan::bench]
fn str_like_middle_sorted(bencher: Bencher) { run_like(bencher, true, "%abc%"); }

// ---------------------------------------------------------------------------
// min/max aggregate
// ---------------------------------------------------------------------------

fn run_minmax<T>(bencher: Bencher, sorted: bool)
where
    T: vortex_array::dtype::NativePType,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    let dict = primitive_dict::<T>(sorted);
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| min_max(&d, &mut ctx).unwrap());
}
#[divan::bench]
fn i64_minmax_plain(bencher: Bencher) { run_minmax::<i64>(bencher, false); }
#[divan::bench]
fn i64_minmax_sorted(bencher: Bencher) { run_minmax::<i64>(bencher, true); }
#[divan::bench]
fn f32_minmax_plain(bencher: Bencher) { run_minmax::<f32>(bencher, false); }
#[divan::bench]
fn f32_minmax_sorted(bencher: Bencher) { run_minmax::<f32>(bencher, true); }

// ---------------------------------------------------------------------------
// is_sorted aggregate
// ---------------------------------------------------------------------------

fn run_is_sorted<T>(bencher: Bencher, sorted: bool)
where
    T: vortex_array::dtype::NativePType,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    let dict = primitive_dict::<T>(sorted);
    bencher
        .with_inputs(|| (dict.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(d, mut ctx)| is_sorted(&d, &mut ctx).unwrap());
}
#[divan::bench]
fn i64_is_sorted_plain(bencher: Bencher) { run_is_sorted::<i64>(bencher, false); }
#[divan::bench]
fn i64_is_sorted_sorted(bencher: Bencher) { run_is_sorted::<i64>(bencher, true); }

// Hush divan warnings about unused canonical import.
const _: fn() = || {
    let _ = size_of::<Canonical>();
    let _ = size_of::<DictArray>();
    let _ = size_of::<PrimitiveArray>();
};
