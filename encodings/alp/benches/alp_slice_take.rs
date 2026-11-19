// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks a take of a few rows of a slice of an ALP array with patches.
//!
//! A file scan gives each column as a slice of a larger encoded segment. A random batch of rows
//! takes a few rows of each slice, so the cost of the slice itself can be more than the take.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    vortex_alp::initialize(&session);
    session
});

/// Rows of the encoded segment.
const SEGMENT_ROWS: usize = 1 << 18;

/// Rows of the slice, a quarter of the segment, as a file scan splits it.
const SLICE_ROWS: usize = SEGMENT_ROWS / 4;

/// Rows taken from the slice.
const TAKEN_ROWS: &[usize] = &[1, 1024];

/// An ALP segment of decimal values, with about 8% of outliers that become patches.
fn alp_segment() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);

    let values: Buffer<f32> = (0..SEGMENT_ROWS)
        .map(|_| {
            if rng.random_range(0..12) == 0 {
                rng.random_range(-1.0e6f32..1.0e6)
            } else {
                f32::from(rng.random_range(0u16..1000)) / 100.0
            }
        })
        .collect();

    let primitive = PrimitiveArray::new(values, Validity::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();

    alp_encode(primitive.as_view(), None, &mut ctx)
        .unwrap()
        .into_array()
}

/// Takes `taken` random rows of the second quarter of the segment.
#[divan::bench(args = TAKEN_ROWS)]
fn slice_take(bencher: Bencher, taken: usize) {
    let slice = alp_segment().slice(SLICE_ROWS..2 * SLICE_ROWS).unwrap();

    let mut rng = StdRng::seed_from_u64(1);
    let mut indices: Vec<u32> = (0..taken)
        .map(|_| rng.random_range(0..SLICE_ROWS as u32))
        .collect();
    indices.sort_unstable();

    let indices = indices.into_iter().collect::<Buffer<u32>>().into_array();

    bencher
        .with_inputs(|| {
            (
                slice.clone(),
                indices.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(slice, indices, mut ctx)| {
            slice
                .take(indices)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}
