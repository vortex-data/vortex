// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures selective filtering around the sparse extraction thresholds.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const NUM_ARRAY_CHUNKS: usize = 64;
// Keep the array density below the outer full-decode policy.
const NUM_SELECTED_CHUNKS: usize = 8;
const CHUNK_SIZE: usize = 1_024;
const LEN: usize = NUM_ARRAY_CHUNKS * CHUNK_SIZE;

// Selection counts around the sparse extraction threshold for 16-bit values.
const BIT_WIDTH: u8 = 16;

fn fixture(selected_per_chunk: usize) -> (ArrayRef, Mask) {
    let values: BufferMut<u32> = (0..LEN)
        .map(|index| (index % (1 << BIT_WIDTH)) as u32)
        .collect();
    let packed = BitPackedData::encode(
        &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        BIT_WIDTH,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array();

    let indices = (0..NUM_SELECTED_CHUNKS).flat_map(|chunk| {
        (0..selected_per_chunk)
            .map(move |index| chunk * CHUNK_SIZE + index * CHUNK_SIZE / selected_per_chunk)
    });
    (packed, Mask::from_indices(LEN, indices))
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = [8, 64, 80])]
fn filter(bencher: Bencher, selected_per_chunk: usize) {
    let (packed, mask) = fixture(selected_per_chunk);

    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (mask.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(mask, ctx)| {
            packed
                .filter(mask.clone())
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
        });
}
