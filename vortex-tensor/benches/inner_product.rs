// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for `inner_product` over tensor columns.
//!
//! The arms vary vector width and input nullability. Their names are intended to remain stable
//! across scalar-function implementation changes so CodSpeed can compare them against `develop`.
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
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::scalar_fns::inner_product::InnerProduct;
use vortex_tensor::vector::Vector;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
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

fn vectors(width: usize, seed: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ELEMENTS)
        .map(|i| (((i + seed) % 97) as f64) - 48.0)
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ELEMENTS / width,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn bench_inner_product(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef) {
    let session = vortex_array::array_session();
    bencher
        .counter(ItemsCount::new(lhs.len()))
        .with_inputs(|| {
            (
                InnerProduct::try_new(lhs.clone(), rhs.clone())
                    .unwrap()
                    .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_inner_product(bencher, vectors(width, 0), vectors(width, 31));
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let validity = Validity::from_iter((0..ELEMENTS / width).map(|i| i % 8 != 0));
    let lhs = MaskedArray::try_new(vectors(width, 0), validity)
        .unwrap()
        .into_array();
    bench_inner_product(bencher, lhs, vectors(width, 31));
}
