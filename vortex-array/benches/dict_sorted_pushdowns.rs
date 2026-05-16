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
// lt without bool materialisation.
//
// What the dict path costs (22 µs at 100K, 1024) goes mostly to scanning the
// codes with a SIMD compare and writing the bool buffer. If the column itself
// is sorted (codes increase monotonically along the row axis), we can replace
// the per-row compare with a single binary search and emit a contiguous
// Mask::from_slices — no per-row materialisation at all.
//
//   lt_sorted_dict        : binary-search values for `k`, then `codes < k`
//                           (current sorted-dict pushdown — has bool alloc)
//   lt_sorted_column_range: binary-search the sorted column for `needle`, emit
//                           Mask::from_slices(N, [(0, k)])
//                           (what we'd do if the column were truly sorted)
//   lt_sorted_column_alltrue: needle past max -> Mask::AllTrue (boundary case)
// ---------------------------------------------------------------------------

// Borrow the slice (avoid Vec clone/drop in the timed section).
#[divan::bench]
fn i64_lt_sorted_column_range(bencher: Bencher) {
    let arr: Vec<i64> = (0..N as i64).collect();
    let needle: i64 = (N / 2) as i64;
    bencher.bench(|| {
        let k = arr.partition_point(|&v| v < needle);
        Mask::from_slices(N, vec![(0, k)])
    });
}

#[divan::bench]
fn i64_lt_sorted_column_alltrue(bencher: Bencher) {
    let arr: Vec<i64> = (0..N as i64).collect();
    let needle: i64 = (N + 1) as i64;
    bencher.bench(|| {
        let k = arr.partition_point(|&v| v < needle);
        if k == 0 {
            Mask::AllFalse(N)
        } else if k == N {
            Mask::AllTrue(N)
        } else {
            Mask::from_slices(N, vec![(0, k)])
        }
    });
}

// Binary search alone (no Mask alloc).
#[divan::bench]
fn i64_lt_sorted_column_search_only(bencher: Bencher) {
    let arr: Vec<i64> = (0..N as i64).collect();
    let needle: i64 = (N / 2) as i64;
    bencher.bench(|| arr.partition_point(|&v| v < needle));
}

// ---------------------------------------------------------------------------
// SIMD compare: pack vs sum vs byte-mask materialisation.
//
// All three iterate the same N u16 codes comparing against a scalar. They
// differ only in what they DO with each bool:
//   - pack: shift-and-or 8 results into a byte, write to BitBuffer (12.5KB for 100K)
//   - sum:  cast bool to u64 and accumulate (returns count, 8 bytes)
//   - byte: write 1 byte (0 or 1) per element to Vec<u8> (100KB for 100K)
//
// Each variant operates on the exact same input so the difference is purely
// the materialisation strategy.
// ---------------------------------------------------------------------------

fn make_codes(len: usize) -> Vec<u16> {
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use rand::RngExt;
    let mut rng = StdRng::seed_from_u64(0);
    (0..len).map(|_| rng.random_range(0..1024) as u16).collect()
}

#[divan::bench]
fn cmp_pack_bits(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| {
        // Chunked 8-at-a-time pack: one byte per 8 cmps.
        let chunks = N / 8;
        let tail = N % 8;
        let mut out: Vec<u8> = Vec::with_capacity(chunks + usize::from(tail > 0));
        let mut i = 0;
        for _ in 0..chunks {
            let mut b: u8 = 0;
            for j in 0..8 {
                b |= u8::from(codes[i + j] < needle) << j;
            }
            out.push(b);
            i += 8;
        }
        if tail > 0 {
            let mut b: u8 = 0;
            for j in 0..tail {
                b |= u8::from(codes[i + j] < needle) << j;
            }
            out.push(b);
        }
        out
    });
}

#[divan::bench]
fn cmp_sum_count(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| {
        let mut count: u64 = 0;
        for &c in &codes {
            count += u64::from(c < needle);
        }
        count
    });
}

// Chunked sum: gives the autovectoriser a fixed-width inner loop.
#[divan::bench]
fn cmp_sum_count_chunked(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| {
        codes
            .chunks_exact(64)
            .map(|c| c.iter().filter(|&&x| x < needle).count() as u64)
            .sum::<u64>()
    });
}

// Iterator-fold variant — sometimes optimises better.
#[divan::bench]
fn cmp_sum_count_iter(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| codes.iter().filter(|&&c| c < needle).count() as u64);
}

#[divan::bench]
fn cmp_byte_mask(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| {
        let mut out: Vec<u8> = Vec::with_capacity(N);
        for &c in &codes {
            out.push(u8::from(c < needle));
        }
        out
    });
}

// Vec<bool> writes 1 byte per element too but goes through bool's
// representation rather than u8 directly — to compare with cmp_byte_mask.
#[divan::bench]
fn cmp_vec_bool(bencher: Bencher) {
    let codes = make_codes(N);
    let needle = 512u16;
    bencher.bench(|| {
        let mut out: Vec<bool> = Vec::with_capacity(N);
        for &c in &codes {
            out.push(c < needle);
        }
        out
    });
}

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
