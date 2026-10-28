// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for `l2_norm` over tensor columns.
//!
//! The arms vary vector width and input nullability. Their names are intended to remain stable
//! across scalar-function implementation changes so CodSpeed can compare them against `develop`.
//!
//! Rows are derived from a fixed element budget rather than fixed per arm, so widening a vector
//! trades rows for elements instead of multiplying the work. See [`ELEMENTS`].

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;
use vortex_tensor::vector::Vector;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

/// Total `f64` elements per operand, held constant across widths: the row count is
/// `ELEMENTS / width`. CodSpeed's CPU simulation charges memory traffic far more than a desktop
/// does, so this budget is what keeps every arm inside the 1 ms per-iteration limit from
/// `docs/developer-guide/benchmarking.md`.
const ELEMENTS: usize = 16_384;

const WIDTHS: &[usize] = &[2, 32, 256];

fn vectors(width: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ELEMENTS).map(|i| ((i % 97) as f64) - 48.0).collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ELEMENTS / width,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn bench_l2_norm(bencher: Bencher, input: ArrayRef) {
    bencher
        .counter(ItemsCount::new(input.len()))
        .with_inputs(|| {
            (
                L2Norm::try_new(input.clone()).unwrap().into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = WIDTHS)]
fn non_nullable(bencher: Bencher, width: usize) {
    bench_l2_norm(bencher, vectors(width));
}

#[divan::bench(args = WIDTHS)]
fn nullable(bencher: Bencher, width: usize) {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter((0..ELEMENTS / width).map(|i| i % 8 != 0));
    let input = MaskedArray::try_new(vectors(width), validity, &mut ctx)
        .unwrap()
        .into_array();
    bench_l2_norm(bencher, input);
}
