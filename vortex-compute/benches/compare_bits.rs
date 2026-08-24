// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the lowest-level comparison kernels: bit-packing a comparison predicate
//! over lanes with [`IndexedSourceExt::map_bits_into`].
//!
//! This is the kernel every native Vortex comparison bottoms out in. `vortex-array`'s
//! `collect_zip_bits` / `collect_bits` — used by the primitive and decimal compare paths —
//! are the allocation plus a `map_bits_into` call, so the shapes here are:
//!
//! - `zip_bits_*` / `const_bits_*`: the kernel alone, writing into caller-owned words.
//!   Array-vs-array via [`LaneZip`], and array-vs-constant over a plain slice.
//! - `collect_zip_bits_*` / `collect_bits_*`: the same kernels with the `BufferMut<u64>`
//!   allocation and [`BitBuffer`] freeze around them, mirroring `vortex-array` exactly.
//!
//! The `zip_bits_*` and `const_bits_*` benchmarks carry `#[cpu_features]`, so they are
//! measured on every walltime CPU-feature leg instead of in simulation. They are written once
//! and compiled differently per leg: the kernel is a branch-free lane loop whose whole cost is
//! how well it auto-vectorizes for the build, which is what comparing legs measures. The
//! `collect_*` benchmarks are untagged and stay in the sharded simulation job, where the
//! allocate-and-freeze wrapper is the part worth watching for instruction-count regressions.
//!
//! Integer lanes compare with their natural ordering; float lanes use `f64::total_cmp`, which
//! is the total ordering `vortex-array`'s `NativePType::is_lt` and friends are built on. The
//! `i128` lanes stand in for decimal comparison, which uses the same kernel.

use divan::Bencher;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::LaneZip;

fn main() {
    divan::main();
}

/// Sized to keep CodSpeed simulation under 1ms per benchmark, matching
/// `vortex-array`'s `compare` benchmarks.
const SIZES: &[usize] = &[8_192];

/// Two operand buffers plus a scalar operand, all drawn from one seeded RNG.
struct Fixture<T> {
    lhs: Buffer<T>,
    rhs: Buffer<T>,
    constant: T,
}

fn fixture<T, F>(n: usize, mut sample: F) -> Fixture<T>
where
    F: FnMut(&mut StdRng) -> T,
{
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let lhs = (0..n).map(|_| sample(&mut rng)).collect::<Buffer<T>>();
    let rhs = (0..n).map(|_| sample(&mut rng)).collect::<Buffer<T>>();
    let constant = sample(&mut rng);
    Fixture { lhs, rhs, constant }
}

fn i32_fixture(n: usize) -> Fixture<i32> {
    fixture(n, |rng| rng.random_range(0i32..100_000_000))
}

fn i64_fixture(n: usize) -> Fixture<i64> {
    fixture(n, |rng| rng.random_range(0i64..100_000_000))
}

fn f64_fixture(n: usize) -> Fixture<f64> {
    fixture(n, |rng| rng.random_range(0.0f64..1.0))
}

fn i128_fixture(n: usize) -> Fixture<i128> {
    fixture(n, |rng| rng.random_range(0i128..100_000_000))
}

fn words(n: usize) -> Vec<u64> {
    vec![0u64; n.div_ceil(64)]
}

/// Bit-pack `f(lhs[i], rhs[i])` into caller-owned words — the array-vs-array kernel.
fn bench_zip<T: Copy + Sync>(
    bencher: Bencher,
    n: usize,
    f: Fixture<T>,
    predicate: impl Fn(T, T) -> bool + Sync,
) {
    bencher.with_inputs(|| words(n)).bench_refs(|out| {
        LaneZip::new(f.lhs.as_slice(), f.rhs.as_slice())
            .map_bits_into(out.as_mut_slice(), |(a, b)| predicate(a, b));
    });
}

/// Bit-pack `f(lhs[i], constant)` into caller-owned words — the array-vs-constant kernel.
fn bench_const<T: Copy + Sync>(
    bencher: Bencher,
    n: usize,
    f: Fixture<T>,
    predicate: impl Fn(T, T) -> bool + Sync,
) {
    bencher.with_inputs(|| words(n)).bench_refs(|out| {
        f.lhs
            .as_slice()
            .map_bits_into(out.as_mut_slice(), |a| predicate(a, f.constant));
    });
}

/// `vortex-array`'s `collect_zip_bits`: allocate the words, run the kernel, freeze the bits.
fn collect_zip_bits<T: Copy>(lhs: &[T], rhs: &[T], predicate: impl Fn(T, T) -> bool) -> BitBuffer {
    let len = lhs.len();
    let mut words = BufferMut::<u64>::zeroed(len.div_ceil(64));
    LaneZip::new(lhs, rhs).map_bits_into(words.as_mut_slice(), |(a, b)| predicate(a, b));
    bit_buffer_from_words(words, len)
}

/// `vortex-array`'s `collect_bits`, the array-vs-constant counterpart.
fn collect_bits<T: Copy>(values: &[T], predicate: impl Fn(T) -> bool) -> BitBuffer {
    let len = values.len();
    let mut words = BufferMut::<u64>::zeroed(len.div_ceil(64));
    values.map_bits_into(words.as_mut_slice(), predicate);
    bit_buffer_from_words(words, len)
}

fn bit_buffer_from_words(words: BufferMut<u64>, len: usize) -> BitBuffer {
    let mut bytes = words.into_byte_buffer();
    bytes.truncate(len.div_ceil(8));
    BitBuffer::new(bytes.freeze(), len)
}

// -----------------------------------------------------------------------------
// Kernel benchmarks, measured per CPU-feature leg.
// -----------------------------------------------------------------------------

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn zip_bits_i32_gte(bencher: Bencher, n: usize) {
    bench_zip(bencher, n, i32_fixture(n), |a, b| a >= b);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn zip_bits_i64_gte(bencher: Bencher, n: usize) {
    bench_zip(bencher, n, i64_fixture(n), |a, b| a >= b);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn zip_bits_i64_eq(bencher: Bencher, n: usize) {
    bench_zip(bencher, n, i64_fixture(n), |a, b| a == b);
}

/// Float lanes under Vortex's total ordering, which is what makes this kernel harder to
/// vectorize than the integer ones.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn zip_bits_f64_lt(bencher: Bencher, n: usize) {
    bench_zip(bencher, n, f64_fixture(n), |a: f64, b: f64| {
        a.total_cmp(&b).is_lt()
    });
}

/// `i128` lanes: the decimal compare path runs the same kernel over double-width lanes.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn zip_bits_i128_gte(bencher: Bencher, n: usize) {
    bench_zip(bencher, n, i128_fixture(n), |a, b| a >= b);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn const_bits_i64_gte(bencher: Bencher, n: usize) {
    bench_const(bencher, n, i64_fixture(n), |a, b| a >= b);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = SIZES)]
fn const_bits_f64_lt(bencher: Bencher, n: usize) {
    bench_const(bencher, n, f64_fixture(n), |a: f64, b: f64| {
        a.total_cmp(&b).is_lt()
    });
}

// -----------------------------------------------------------------------------
// Allocate-and-freeze wrappers, measured in simulation.
// -----------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn collect_zip_bits_i64_gte(bencher: Bencher, n: usize) {
    let f = i64_fixture(n);
    bencher.bench(|| collect_zip_bits(f.lhs.as_slice(), f.rhs.as_slice(), |a, b| a >= b));
}

#[divan::bench(args = SIZES)]
fn collect_zip_bits_f64_lt(bencher: Bencher, n: usize) {
    let f = f64_fixture(n);
    bencher.bench(|| {
        collect_zip_bits(f.lhs.as_slice(), f.rhs.as_slice(), |a: f64, b: f64| {
            a.total_cmp(&b).is_lt()
        })
    });
}

#[divan::bench(args = SIZES)]
fn collect_bits_i64_gte(bencher: Bencher, n: usize) {
    let f = i64_fixture(n);
    bencher.bench(|| collect_bits(f.lhs.as_slice(), |a| a >= f.constant));
}
