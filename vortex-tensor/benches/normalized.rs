// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for decoding the `Normalized` encoding over tensor columns.
//!
//! The arms vary vector width and input nullability. Their names are intended to remain stable
//! across implementation changes so CodSpeed can compare them against `develop`.
//!
//! Rows are derived from a fixed element budget rather than fixed per arm, so widening a vector
//! trades rows for elements instead of multiplying the work. See [`ELEMENTS`].

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
use vortex_tensor::vector::Vector;

// Decoding allocates the output inside the timed region, so use the vendored allocator instead
// of measuring glibc differences between CodSpeed runner images.
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

fn norms(rows: usize) -> ArrayRef {
    let values: Buffer<f64> = (0..rows).map(|i| 1.0 + ((i % 13) as f64) / 13.0).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

fn bench_normalized(bencher: Bencher, normalized: ArrayRef) {
    let session = vortex_array::array_session();
    let rows = normalized.len();
    let norms = norms(rows);
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| {
            let mut ctx = session.create_execution_ctx();
            let array = Normalized::try_new(normalized.clone(), norms.clone(), &mut ctx).unwrap();
            (array, ctx)
        })
        .bench_values(|(array, mut ctx)| {
            array
                .into_array()
                .execute::<ExtensionArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_normalized(bencher, normalized_vectors(width));
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let validity = Validity::from_iter((0..ELEMENTS / width).map(|i| i % 8 != 0));
    let normalized = MaskedArray::try_new(normalized_vectors(width), validity)
        .unwrap()
        .into_array();
    bench_normalized(bencher, normalized);
}
