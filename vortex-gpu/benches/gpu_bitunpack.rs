// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use cudarc::driver::CudaContext;
use divan::Bencher;
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::BitPackedArray;
use vortex_gpu::cuda_bit_unpack;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Data sizes: 1GB, 2.5GB, 5GB, 10GB
// These are approximate sizes in bytes, accounting for bit-packing compression
const DATA_SIZES: &[(usize, &str)] = &[
    (268_435_456, "1GB"),    // ~1GB when unpacked (268M * 4 bytes)
    (671_088_640, "2.5GB"),  // ~2.5GB when unpacked
    (1_342_177_280, "5GB"),  // ~5GB when unpacked
    (2_684_354_560, "10GB"), // ~10GB when unpacked
];

/// Creates a bitpackable dataset of the given size.
/// Values are chosen to fit in 6 bits (0-63) to ensure no patches are needed.
fn make_bitpackable_array<T: NativePType>(len: usize) -> BitPackedArray {
    let mut rng = StdRng::seed_from_u64(42);
    // Generate values that fit in 6 bits (0-63)
    let values = (0..len)
        .map(|_| T::from(rng.random_range(0..64)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();

    // Encode with 6-bit width, which will not need patches
    BitPackedArray::encode(values.as_ref(), 6).unwrap()
}

// #[divan::bench(types = [u32, u64], args = DATA_SIZES)]
#[divan::bench(types = [u32], args = DATA_SIZES)]
fn gpu_decompress<T: NativePType>(bencher: Bencher, (len, _label): (usize, &str)) {
    // Round up to next multiple of 1024 (GPU kernel requirement)
    let len = len.next_multiple_of(1024);
    let array = make_bitpackable_array::<T>(len);

    // Initialize CUDA context once
    let ctx = CudaContext::new(0).unwrap();
    ctx.set_blocking_synchronize().unwrap();
    let ctx = Arc::new(ctx);

    bencher
        .counter(BytesCount::of_many::<T>(len))
        .with_inputs(|| array.clone())
        .bench_values(|array| cuda_bit_unpack(&array, Arc::clone(&ctx)).unwrap());
}

// #[divan::bench(types = [u32, u64], args = DATA_SIZES)]
#[divan::bench(types = [u32], args = DATA_SIZES)]
fn cpu_canonicalize<T: NativePType>(bencher: Bencher, (len, _label): (usize, &str)) {
    // Round up to next multiple of 1024 for fair comparison
    let len = len.next_multiple_of(1024);
    let array = make_bitpackable_array::<T>(len);

    bencher
        .counter(BytesCount::of_many::<T>(len))
        .with_inputs(|| array.clone())
        .bench_values(|array| array.into_array().to_canonical());
}
