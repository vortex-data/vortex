// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::take_bits;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
        let _ = is_x86_feature_detected!("avx512bw");
    }

    divan::main();
}

const INDEX_COUNT: &[usize] = &[65_536, 1_048_576];

fn make_bits(len: usize) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(42);
    BitBuffer::collect_bool(len, |_| rng.random())
}

fn make_4096_u16(count: usize) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(43);
    (0..count).map(|_| rng.random_range(0..4096u16)).collect()
}

fn make_65536_u16(count: usize) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(44);
    (0..count).map(|_| rng.random()).collect()
}

fn make_1m_u32(count: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(45);
    (0..count)
        .map(|_| rng.random_range(0..1u32 << 20))
        .collect()
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDEX_COUNT)]
fn take_u16_from_4096(bencher: Bencher, count: usize) {
    let bits = make_bits(4096);
    let indices = make_4096_u16(count);
    bencher.bench(|| take_bits(bits.as_view(), &indices, None));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDEX_COUNT)]
fn take_u16_from_65536(bencher: Bencher, count: usize) {
    let bits = make_bits(65_536);
    let indices = make_65536_u16(count);
    bencher.bench(|| take_bits(bits.as_view(), &indices, None));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDEX_COUNT)]
fn take_u32_from_1m(bencher: Bencher, count: usize) {
    let bits = make_bits(1 << 20);
    let indices = make_1m_u32(count);
    bencher.bench(|| take_bits(bits.as_view(), &indices, None));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDEX_COUNT)]
fn take_u16_single_zero(bencher: Bencher, count: usize) {
    let bits = BitBuffer::collect_bool(65_536, |i| i != 12_345);
    let indices = make_65536_u16(count);
    bencher.bench(|| take_bits(bits.as_view(), &indices, None));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INDEX_COUNT)]
fn take_u16_nullable_from_65536(bencher: Bencher, count: usize) {
    let bits = make_bits(65_536);
    let indices = make_65536_u16(count);
    let validity = make_bits(count);
    bencher.bench(|| take_bits(bits.as_view(), &indices, Some(validity.as_view())));
}
