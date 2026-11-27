// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take benchmarks for [`ZstdArray`].
//!
//! A frame is the smallest independently decompressible unit, so what a take costs is set by how
//! many frames the requested rows land in rather than by how many rows are asked for. The index
//! shapes below span that axis: requests that cluster into a few frames, and requests that reach
//! every frame and so cannot avoid a full decompress. `OutOfOrder` covers the shape no mask can
//! name, which has to keep working through the default gather.
//!
//! [`ZstdArray`]: vortex_zstd::ZstdArray

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const NUM_VALUES: usize = 262_144;
/// Matches `vortex_btrblocks::schemes::string::ZstdScheme`, giving 32 frames.
const VALUES_PER_FRAME: usize = 8_192;

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Selection clusters into the leading frames; the rest are never inflated.
    LeadingTenth,
    /// Selection clusters into two separate groups of frames.
    TwoClusters,
    /// One run of rows, which slices out the frames around it.
    Contiguous,
    /// Selection reaches every frame, so a full decompress is unavoidable.
    Spread,
    /// Dense enough that every frame is needed and most rows are asked for.
    Dense,
    /// The same rows as a clustered request but descending, which no mask can name.
    OutOfOrder,
}

const SHAPES: &[Shape] = &[
    Shape::LeadingTenth,
    Shape::TwoClusters,
    Shape::Contiguous,
    Shape::Spread,
    Shape::Dense,
    Shape::OutOfOrder,
];

fn selected(shape: Shape, row: usize) -> bool {
    match shape {
        Shape::LeadingTenth => row < NUM_VALUES / 10 && row.is_multiple_of(8),
        Shape::TwoClusters => {
            (row < NUM_VALUES / 50 || (NUM_VALUES * 3 / 5..NUM_VALUES * 62 / 100).contains(&row))
                && row.is_multiple_of(4)
        }
        Shape::Contiguous => (NUM_VALUES / 4..NUM_VALUES * 3 / 4).contains(&row),
        Shape::Spread => {
            // A cheap hash so the selection is spread without pulling in an rng.
            let mut state = row as u64 ^ 0x9e37_79b9_7f4a_7c15;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.is_multiple_of(100)
        }
        Shape::Dense => !row.is_multiple_of(20),
        Shape::OutOfOrder => row.is_multiple_of(64),
    }
}

/// The rows `shape` selects, as the ascending index list a take expects.
fn indices(shape: Shape) -> ArrayRef {
    let mut rows: Vec<u64> = (0..NUM_VALUES)
        .filter(|&row| selected(shape, row))
        .map(|row| row as u64)
        .collect();
    if matches!(shape, Shape::OutOfOrder) {
        // Reversed, so the indices name the same rows but cannot be handed over as a mask.
        rows.reverse();
    }
    PrimitiveArray::new(Buffer::<u64>::from(rows), Validity::NonNullable).into_array()
}

/// Values compressible enough that skipping a frame is worth the bookkeeping.
fn values() -> Vec<String> {
    (0..NUM_VALUES)
        .map(|index| {
            format!(
                "session-{index:07}-{}-region-{}",
                ["alpha", "beta", "gamma", "delta", "epsilon"][index % 5],
                ["us-east-1", "us-west-2", "eu-central-1"][index % 3]
            )
        })
        .collect()
}

fn compressed() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let vbv = VarBinViewArray::from_iter_str(values());
    Zstd::from_var_bin_view_without_dict(&vbv, 3, VALUES_PER_FRAME, &mut ctx)
        .unwrap()
        .into_array()
}

fn bench_take(bencher: Bencher, array: ArrayRef, shape: Shape) {
    let indices = indices(shape);
    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| indices.clone())
        .bench_values(|indices| {
            let mut ctx = SESSION.create_execution_ctx();
            array
                .take(indices)
                .unwrap()
                .execute::<VarBinViewArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = SHAPES)]
fn zstd(bencher: Bencher, shape: Shape) {
    bench_take(bencher, compressed(), shape);
}

/// The same rows uncompressed, so the decompression a shape drives can be told apart from the cost
/// of the gather.
#[divan::bench(args = SHAPES)]
fn varbinview(bencher: Bencher, shape: Shape) {
    bench_take(
        bencher,
        VarBinViewArray::from_iter_str(values()).into_array(),
        shape,
    );
}
