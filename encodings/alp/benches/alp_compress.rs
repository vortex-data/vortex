// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_alp::ALPFloat;
use vortex_alp::ALPRDFloat;
use vortex_alp::Exponents;
use vortex_alp::RDEncoder;
use vortex_alp::alp_encode;
use vortex_alp::decompress_into_array;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;

fn main() {
    warm_up_vtables();
    divan::main();
}

// ---------------------------------------------------------------------------
// Classic ALP benchmarks
// ---------------------------------------------------------------------------

const BENCH_ARGS: &[(usize, f64, f64)] = &[
    // length, fraction_patch, fraction_valid
    (1_000, 0.0, 0.25),
    (1_000, 0.01, 0.25),
    (1_000, 0.1, 0.25),
    (1_000, 0.0, 0.95),
    (1_000, 0.01, 0.95),
    (1_000, 0.1, 0.95),
    (1_000, 0.0, 1.0),
    (1_000, 0.01, 1.0),
    (1_000, 0.1, 1.0),
    (10_000, 0.0, 0.25),
    (10_000, 0.01, 0.25),
    (10_000, 0.1, 0.25),
    (10_000, 0.0, 0.95),
    (10_000, 0.01, 0.95),
    (10_000, 0.1, 0.95),
    (10_000, 0.0, 1.0),
    (10_000, 0.01, 1.0),
    (10_000, 0.1, 1.0),
];

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn compress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();
    let array = PrimitiveArray::new(values, validity);

    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, None).unwrap())
}

#[divan::bench(types = [f32, f64], args = BENCH_ARGS)]
fn decompress_alp<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let validity = if fraction_valid < 1.0 {
        Validity::from_iter((0..values.len()).map(|_| rng.random_bool(fraction_valid)))
    } else {
        Validity::NonNullable
    };
    let values = values.freeze();
    bencher
        .with_inputs(|| {
            (
                alp_encode(
                    &PrimitiveArray::new(Buffer::copy_from(&values), validity.clone()),
                    None,
                )
                .unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

// ---------------------------------------------------------------------------
// Large-scale ALP benchmarks (100K and 1M elements)
// ---------------------------------------------------------------------------

const LARGE_BENCH_ARGS: &[(usize, f64, f64)] = &[
    // length, fraction_patch, fraction_valid
    (100_000, 0.0, 1.0),
    (100_000, 0.01, 1.0),
    (100_000, 0.1, 1.0),
    (1_000_000, 0.0, 1.0),
    (1_000_000, 0.01, 1.0),
    (1_000_000, 0.1, 1.0),
];

#[divan::bench(types = [f32, f64], args = LARGE_BENCH_ARGS)]
fn compress_alp_large<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, _fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(42);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let values = values.freeze();
    let array = PrimitiveArray::new(values, Validity::NonNullable);

    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, None).unwrap())
}

#[divan::bench(types = [f32, f64], args = LARGE_BENCH_ARGS)]
fn decompress_alp_large<T: ALPFloat + NativePType>(bencher: Bencher, args: (usize, f64, f64)) {
    let (n, fraction_patch, _fraction_valid) = args;
    let mut rng = StdRng::seed_from_u64(42);
    let mut values = buffer![T::from(1.234).unwrap(); n].into_mut();
    if fraction_patch > 0.0 {
        for index in 0..values.len() {
            if rng.random_bool(fraction_patch) {
                values[index] = T::from(1000.0).unwrap()
            }
        }
    }
    let values = values.freeze();

    bencher
        .with_inputs(|| {
            (
                alp_encode(
                    &PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable),
                    None,
                )
                .unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

// ---------------------------------------------------------------------------
// ALP with pre-computed exponents (measures encoding throughput sans search)
// ---------------------------------------------------------------------------

const PRECOMPUTED_EXP_ARGS: &[usize] = &[10_000, 100_000, 1_000_000];

#[divan::bench(types = [f32, f64], args = PRECOMPUTED_EXP_ARGS)]
fn compress_alp_precomputed_exp<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = buffer![T::from(1.234).unwrap(); n];
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    let exponents = Exponents { e: 9, f: 6 };

    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, Some(exponents)).unwrap())
}

// ---------------------------------------------------------------------------
// ALP with varied data distributions
// ---------------------------------------------------------------------------

/// Generate realistic financial-style price data (e.g. $10.00 - $999.99).
fn generate_price_data<T: ALPFloat + NativePType>(n: usize) -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(123);
    let mut buf = vortex_buffer::BufferMut::<T>::with_capacity(n);
    for _ in 0..n {
        let dollars: f64 = rng.random_range(10.0..1000.0);
        // Round to 2 decimal places
        let price = (dollars * 100.0).round() / 100.0;
        buf.push(T::from(price).unwrap());
    }
    buf.freeze()
}

/// Generate scientific measurement data (small values with varying precision).
fn generate_scientific_data<T: ALPFloat + NativePType>(n: usize) -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(456);
    let mut buf = vortex_buffer::BufferMut::<T>::with_capacity(n);
    for _ in 0..n {
        let base: f64 = rng.random_range(0.001..10.0);
        let val = (base * 1_000_000.0).round() / 1_000_000.0;
        buf.push(T::from(val).unwrap());
    }
    buf.freeze()
}

/// Generate sensor/temperature-like data (narrow range, moderate precision).
fn generate_sensor_data<T: ALPFloat + NativePType>(n: usize) -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(789);
    let mut buf = vortex_buffer::BufferMut::<T>::with_capacity(n);
    for _ in 0..n {
        let temp: f64 = rng.random_range(20.0..25.0);
        let val = (temp * 100.0).round() / 100.0;
        buf.push(T::from(val).unwrap());
    }
    buf.freeze()
}

const DISTRIBUTION_SIZES: &[usize] = &[10_000, 100_000];

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn compress_alp_price_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_price_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, None).unwrap())
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn decompress_alp_price_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_price_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| {
            (
                alp_encode(&array, None).unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn compress_alp_scientific_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_scientific_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, None).unwrap())
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn decompress_alp_scientific_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_scientific_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| {
            (
                alp_encode(&array, None).unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn compress_alp_sensor_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_sensor_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| &array)
        .bench_values(|array| alp_encode(array, None).unwrap())
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn decompress_alp_sensor_data<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_sensor_data::<T>(n);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    bencher
        .with_inputs(|| {
            (
                alp_encode(&array, None).unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(v, mut ctx)| decompress_into_array(v, &mut ctx));
}

// ---------------------------------------------------------------------------
// ALP-RD benchmarks
// ---------------------------------------------------------------------------

#[divan::bench(types = [f32, f64], args = [10_000, 100_000])]
fn compress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);

    bencher
        .with_inputs(|| (&primitive, &encoder))
        .bench_refs(|(primitive, encoder)| encoder.encode(primitive))
}

#[divan::bench(types = [f32, f64], args = [10_000, 100_000])]
fn decompress_rd<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    let encoded = encoder.encode(&primitive);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

// ---------------------------------------------------------------------------
// ALP-RD large-scale benchmarks
// ---------------------------------------------------------------------------

const RD_LARGE_ARGS: &[usize] = &[100_000, 1_000_000];

#[divan::bench(types = [f32, f64], args = RD_LARGE_ARGS)]
fn compress_rd_large<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);

    bencher
        .with_inputs(|| (&primitive, &encoder))
        .bench_refs(|(primitive, encoder)| encoder.encode(primitive))
}

#[divan::bench(types = [f32, f64], args = RD_LARGE_ARGS)]
fn decompress_rd_large<T: ALPRDFloat>(bencher: Bencher, n: usize) {
    let primitive = PrimitiveArray::new(buffer![T::from(1.23).unwrap(); n], Validity::NonNullable);
    let encoder = RDEncoder::new(&[T::from(1.23).unwrap()]);
    let encoded = encoder.encode(&primitive);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

// ---------------------------------------------------------------------------
// ALP-RD with varied data (realistic doubles that need RD encoding)
// ---------------------------------------------------------------------------

/// Generate data that exercises ALP-RD well: full-precision doubles.
fn generate_rd_varied_data<T: ALPRDFloat + NativePType>(n: usize) -> Buffer<T> {
    let mut rng = StdRng::seed_from_u64(999);
    let mut buf = vortex_buffer::BufferMut::<T>::with_capacity(n);
    for _ in 0..n {
        let v: f64 = rng.random_range(1.0..100.0);
        buf.push(T::from(v).unwrap());
    }
    buf.freeze()
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn compress_rd_varied<T: ALPRDFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_rd_varied_data::<T>(n);
    let sample_vals: Vec<T> = values.iter().copied().take(64).collect();
    let encoder = RDEncoder::new(&sample_vals);
    let array = PrimitiveArray::new(values, Validity::NonNullable);

    bencher
        .with_inputs(|| (&array, &encoder))
        .bench_refs(|(array, encoder)| encoder.encode(array))
}

#[divan::bench(types = [f32, f64], args = DISTRIBUTION_SIZES)]
fn decompress_rd_varied<T: ALPRDFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_rd_varied_data::<T>(n);
    let sample_vals: Vec<T> = values.iter().copied().take(64).collect();
    let encoder = RDEncoder::new(&sample_vals);
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    let encoded = encoder.encode(&array);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

// ---------------------------------------------------------------------------
// ALP exponent finding benchmark (measures parameter search cost)
// ---------------------------------------------------------------------------

const EXPONENT_SEARCH_ARGS: &[usize] = &[100, 1_000, 10_000];

#[divan::bench(types = [f32, f64], args = EXPONENT_SEARCH_ARGS)]
fn find_best_exponents<T: ALPFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_price_data::<T>(n);
    bencher
        .with_inputs(|| values.as_ref())
        .bench_values(|values| T::find_best_exponents(values))
}

// ---------------------------------------------------------------------------
// ALP-RD dictionary building benchmark
// ---------------------------------------------------------------------------

#[divan::bench(types = [f32, f64], args = EXPONENT_SEARCH_ARGS)]
fn rd_encoder_new<T: ALPRDFloat + NativePType>(bencher: Bencher, n: usize) {
    let values = generate_rd_varied_data::<T>(n);
    let sample: Vec<T> = values.iter().copied().collect();
    bencher
        .with_inputs(|| sample.as_slice())
        .bench_values(|sample| RDEncoder::new(sample))
}
