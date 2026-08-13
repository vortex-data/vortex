// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for encoding and decoding `Normalized` tensor columns.
//!
//! The arms keep the element count fixed while varying vector width and nullability. Their stable
//! names let CodSpeed compare implementation changes against `develop`.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::encodings::normalized::Normalized;
use vortex_tensor::encodings::normalized::normalize;
use vortex_tensor::vector::Vector;

// Both directions allocate their output inside the timed region, so use the vendored allocator
// instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Total `f64` elements per operand, held constant across widths: the row count is
/// `ELEMENTS / width`. CodSpeed's CPU simulation charges memory traffic far more than a desktop
/// does, so this budget is what keeps every arm inside the 1 ms per-iteration limit from
/// `docs/developer-guide/benchmarking.md`.
const ELEMENTS: usize = 8_192;

const WIDTHS: &[usize] = &[2, 32, 256];

fn normalized_vectors(width: usize) -> ArrayRef {
    let value = 1.0 / (width as f64).sqrt();
    let elements: Buffer<f64> = (0..ELEMENTS).map(|_| value).collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ELEMENTS / width,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn plain_vectors(width: usize) -> ArrayRef {
    let rows = ELEMENTS / width;
    // Keep every row nonzero so encoding always takes the division path.
    let elements: Buffer<f64> = (0..rows)
        .flat_map(|row| (0..width).map(move |i| (row + 1) as f64 * (i + 1) as f64))
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        rows,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn sparse_nulls(width: usize) -> Validity {
    Validity::from_iter((0..ELEMENTS / width).map(|i| !i.is_multiple_of(8)))
}

fn norms(rows: usize) -> ArrayRef {
    let values: Buffer<f64> = (0..rows).map(|i| 1.0 + ((i % 13) as f64) / 13.0).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

fn bench_decode(bencher: Bencher, normalized: ArrayRef, validity: Validity) {
    let session = vortex_array::array_session();
    let rows = normalized.len();
    let norms = norms(rows);
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| {
            let mut ctx = session.create_execution_ctx();
            let array = Normalized::try_new(
                normalized.clone(),
                norms.clone(),
                validity.clone(),
                &mut ctx,
            )
            .unwrap();
            (array, ctx)
        })
        .bench_values(|(array, mut ctx)| {
            array
                .into_array()
                .execute::<ExtensionArray>(&mut ctx)
                .unwrap()
        });
}

fn bench_encode(bencher: Bencher, input: ArrayRef) {
    let session = vortex_array::array_session();
    let rows = input.len();
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| (input.clone(), session.create_execution_ctx()))
        .bench_values(|(input, mut ctx)| normalize(input, &mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_decode(bencher, normalized_vectors(width), Validity::NonNullable);
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    bench_decode(bencher, normalized_vectors(width), sparse_nulls(width));
}

#[divan::bench(args = WIDTHS)]
fn encode_non_nullable(bencher: Bencher, width: usize) {
    bench_encode(bencher, plain_vectors(width));
}

#[divan::bench(args = WIDTHS)]
fn encode_nullable(bencher: Bencher, width: usize) {
    let input = MaskedArray::try_new(plain_vectors(width), sparse_nulls(width))
        .unwrap()
        .into_array();

    bench_encode(bencher, input);
}
