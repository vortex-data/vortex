// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small-dictionary take compared with scalar and generic SIMD implementations.

#![allow(unexpected_cfgs)]
#![cfg_attr(vortex_nightly_portable_simd, feature(portable_simd))]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::test_harness::take_values_fallback_u8;
use vortex_array::test_harness::take_values_fallback_u16;
use vortex_array::test_harness::take_values_fallback_u32;
use vortex_array::test_harness::take_values_fallback_u64;
use vortex_array::test_harness::take_values_u8;
use vortex_array::test_harness::take_values_u16;
use vortex_array::test_harness::take_values_u32;
use vortex_array::test_harness::take_values_u64;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 1_048_576;
static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn codes<const NUM_VALUES: usize>() -> Vec<u8> {
    assert!(NUM_VALUES.is_power_of_two());
    assert!(NUM_VALUES <= 256);
    (0..LEN)
        .map(|index| ((index.wrapping_mul(37) ^ (index >> 3)) & (NUM_VALUES - 1)) as u8)
        .collect()
}

fn codes_with_len<const NUM_VALUES: usize>(len: usize) -> Vec<u8> {
    assert!(NUM_VALUES.is_power_of_two());
    (0..len)
        .map(|index| ((index.wrapping_mul(37) ^ (index >> 3)) & (NUM_VALUES - 1)) as u8)
        .collect()
}

fn values<const NUM_VALUES: usize>() -> [u8; 256] {
    let mut values = [0; 256];
    for (index, value) in values[..NUM_VALUES].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    values
}

#[inline(never)]
fn take_scalar(values: &[u8], codes: &[u8], output: &mut [u8]) {
    for (output, &code) in output.iter_mut().zip(codes) {
        *output = values[usize::from(code)];
    }
}

#[divan::bench(consts = [2usize, 4, 8, 16, 32, 64, 128, 256])]
fn scalar<const NUM_VALUES: usize>(bencher: Bencher) {
    let values = values::<NUM_VALUES>();
    let codes = codes::<NUM_VALUES>();
    let mut output = vec![0; LEN];
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        take_scalar(
            black_box(&values),
            black_box(&codes),
            black_box(&mut output),
        );
        black_box(&output);
    });
}

#[divan::bench(consts = [2usize, 4, 8, 16, 32, 64, 128, 256])]
fn vortex_take<const NUM_VALUES: usize>(bencher: Bencher) {
    let values = PrimitiveArray::from_iter(values::<NUM_VALUES>()[..NUM_VALUES].iter().copied())
        .into_array();
    let indices = PrimitiveArray::from_iter(codes::<NUM_VALUES>()).into_array();
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (indices.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(indices, mut ctx)| {
            values
                .take(indices)
                .unwrap()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(consts = [2usize, 4, 8, 16, 32, 64, 128, 256])]
fn vortex_kernel<const NUM_VALUES: usize>(bencher: Bencher) {
    let values = values::<NUM_VALUES>();
    let codes = codes::<NUM_VALUES>();
    let allocator = BufferAllocatorRef::statically_allocated();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        take_values_u8(
            black_box(&values[..NUM_VALUES]),
            black_box(&codes),
            black_box(&allocator),
        )
    });
}

const TAKE_LENGTHS: &[usize] = &[16, 32, 63, 64, 65, 128, 256, 1_024, 16_384];

#[divan::bench(args = TAKE_LENGTHS)]
fn vortex_kernel_by_len(bencher: Bencher, len: usize) {
    let values = values::<32>();
    let codes = codes_with_len::<32>(len);
    let allocator = BufferAllocatorRef::statically_allocated();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        take_values_u8(
            black_box(&values[..32]),
            black_box(&codes),
            black_box(&allocator),
        )
    });
}

#[divan::bench(args = TAKE_LENGTHS)]
fn scalar_kernel_by_len(bencher: Bencher, len: usize) {
    let values = values::<32>();
    let codes = codes_with_len::<32>(len);
    let allocator = BufferAllocatorRef::statically_allocated();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        take_values_fallback_u8(
            black_box(&values[..32]),
            black_box(&codes),
            black_box(&allocator),
        )
    });
}

macro_rules! bench_wide_kernel {
    ($name:ident, $ty:ty, $take:path, [$($num_values:expr),+ $(,)?]) => {
        #[divan::bench(consts = [$($num_values),+])]
        fn $name<const NUM_VALUES: usize>(bencher: Bencher) {
            let values = (0..NUM_VALUES)
                .map(|index| <$ty>::from((index as u8).wrapping_mul(37).wrapping_add(11)))
                .collect::<Vec<_>>();
            let codes = codes::<NUM_VALUES>();
            let allocator = BufferAllocatorRef::statically_allocated();
            bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
                $take(
                    black_box(&values),
                    black_box(&codes),
                    black_box(&allocator),
                )
            });
        }
    };
}

bench_wide_kernel!(
    vortex_kernel_u16,
    u16,
    take_values_u16,
    [2usize, 4, 8, 16, 32, 64, 128]
);
bench_wide_kernel!(
    gather_u16,
    u16,
    take_values_fallback_u16,
    [2usize, 4, 8, 16, 32, 64, 128]
);
bench_wide_kernel!(
    vortex_kernel_u32,
    u32,
    take_values_u32,
    [2usize, 4, 8, 16, 32, 64]
);
bench_wide_kernel!(
    gather_u32,
    u32,
    take_values_fallback_u32,
    [2usize, 4, 8, 16, 32, 64]
);
bench_wide_kernel!(
    vortex_kernel_u64,
    u64,
    take_values_u64,
    [2usize, 4, 8, 16, 32]
);
bench_wide_kernel!(
    gather_u64,
    u64,
    take_values_fallback_u64,
    [2usize, 4, 8, 16, 32]
);

#[divan::bench]
fn allocate_and_copy(bencher: Bencher) {
    let codes = codes::<64>();
    let allocator = BufferAllocatorRef::statically_allocated();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let mut output = BufferMut::with_capacity_in(LEN, black_box(allocator.clone()));
        output.extend_from_slice(black_box(&codes));
        black_box(output.freeze())
    });
}

#[divan::bench]
fn reuse_and_copy(bencher: Bencher) {
    let codes = codes::<64>();
    let mut output = vec![0; LEN];
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        black_box(&mut output).copy_from_slice(black_box(&codes));
        black_box(output[0]);
    });
}

mod fearless {
    use fearless_simd::Level;
    use fearless_simd::Simd;
    use fearless_simd::dispatch;
    use fearless_simd::prelude::*;
    use fearless_simd::u8x32;
    use fearless_simd_macros::simd;

    use super::*;

    #[simd]
    #[inline(never)]
    fn take_kernel<S: Simd, const NUM_VALUES: usize>(
        simd: S,
        values: &[u8; 256],
        codes: &[u8],
        output: &mut [u8],
    ) {
        let table = u8x32::from_slice(simd, &values[..32]);
        let chunks = codes.len() / u8x32::<S>::LEN;
        for chunk in 0..chunks {
            let offset = chunk * u8x32::<S>::LEN;
            let indices = u8x32::from_slice(simd, &codes[offset..offset + u8x32::<S>::LEN]);
            assert!(indices.simd_lt(NUM_VALUES as u8).all_true());
            table
                .swizzle_dyn(indices)
                .store_slice(&mut output[offset..offset + u8x32::<S>::LEN]);
        }
        let tail = chunks * u8x32::<S>::LEN;
        take_scalar(values, &codes[tail..], &mut output[tail..]);
    }

    #[inline(never)]
    fn take<const NUM_VALUES: usize>(
        level: Level,
        values: &[u8; 256],
        codes: &[u8],
        output: &mut [u8],
    ) {
        dispatch!(level, simd => take_kernel::<_, NUM_VALUES>(simd, values, codes, output));
    }

    #[divan::bench(consts = [2usize, 4, 8, 16, 32])]
    fn u8<const NUM_VALUES: usize>(bencher: Bencher) {
        let level = Level::new();
        let values = values::<NUM_VALUES>();
        let codes = codes::<NUM_VALUES>();
        let mut output = vec![0; LEN];
        bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
            take::<NUM_VALUES>(
                black_box(level),
                black_box(&values),
                black_box(&codes),
                black_box(&mut output),
            );
            black_box(&output);
        });
    }
}

#[cfg(vortex_nightly_portable_simd)]
mod portable_simd {
    use std::simd::Simd;
    use std::simd::SimdElement;
    use std::simd::num::SimdUint;

    use super::*;

    #[inline(never)]
    fn take<T, const LANES: usize>(values: &[T], codes: &[u8], output: &mut [T])
    where
        T: Copy + Default + SimdElement,
    {
        let chunks = codes.len() / LANES;
        for chunk in 0..chunks {
            let offset = chunk * LANES;
            let indices = Simd::<u8, LANES>::from_slice(&codes[offset..]).cast::<usize>();
            Simd::<T, LANES>::gather_or_default(values, indices)
                .copy_to_slice(&mut output[offset..]);
        }
        let tail = chunks * LANES;
        for (output, &code) in output[tail..].iter_mut().zip(&codes[tail..]) {
            *output = values[usize::from(code)];
        }
    }

    fn bench<T, const LANES: usize, const NUM_VALUES: usize>(bencher: Bencher)
    where
        T: Copy + Default + From<u8> + SimdElement,
    {
        let values = (0..NUM_VALUES)
            .map(|index| T::from((index as u8).wrapping_mul(37).wrapping_add(11)))
            .collect::<Vec<_>>();
        let codes = codes::<NUM_VALUES>();
        let mut output = vec![T::default(); LEN];
        bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
            take::<T, LANES>(
                black_box(&values),
                black_box(&codes),
                black_box(&mut output),
            );
            black_box(&output);
        });
    }

    #[divan::bench(consts = [2usize, 4, 8, 16, 32])]
    fn u8<const NUM_VALUES: usize>(bencher: Bencher) {
        bench::<u8, 64, NUM_VALUES>(bencher);
    }

    #[divan::bench(consts = [2usize, 4, 8, 16, 32])]
    fn u16<const NUM_VALUES: usize>(bencher: Bencher) {
        bench::<u16, 32, NUM_VALUES>(bencher);
    }

    #[divan::bench(consts = [2usize, 4, 8, 16, 32])]
    fn u32<const NUM_VALUES: usize>(bencher: Bencher) {
        bench::<u32, 16, NUM_VALUES>(bencher);
    }

    #[divan::bench(consts = [2usize, 4, 8, 16, 32])]
    fn u64<const NUM_VALUES: usize>(bencher: Bencher) {
        bench::<u64, 8, NUM_VALUES>(bencher);
    }
}
