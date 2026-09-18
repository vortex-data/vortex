// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);
const ROWS: usize = 1_024;
const CHUNKS: &[usize] = &[2, 32];

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn canonicalize(bencher: Bencher, chunk: ArrayRef, nchunks: usize) {
    let array = ChunkedArray::try_new(
        std::iter::repeat_n(chunk.clone(), nchunks),
        chunk.dtype().clone(),
    )
    .unwrap()
    .into_array();
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx).unwrap());
}

#[divan::bench(args = CHUNKS, consts = [1, 8])]
fn structs<const FIELDS: usize>(bencher: Bencher, nchunks: usize) {
    let field = PrimitiveArray::from_iter(0..ROWS as u64).into_array();
    let chunk = StructArray::try_from_iter(
        (0..FIELDS).map(|idx| (format!("field_{idx}"), field.clone())),
    )
    .unwrap();
    canonicalize(bencher, chunk.into_array(), nchunks);
}

#[divan::bench(args = CHUNKS)]
fn lists(bencher: Bencher, nchunks: usize) {
    let elements = PrimitiveArray::from_iter(0..4 * ROWS as u64).into_array();
    let offsets: Buffer<u64> = (0..=ROWS).map(|idx| 4 * idx as u64).collect();
    let chunk = ListArray::try_new(elements, offsets.into_array(), Validity::NonNullable).unwrap();
    canonicalize(bencher, chunk.into_array(), nchunks);
}

#[divan::bench(args = CHUNKS, consts = [false, true])]
fn listviews<const OVERLAPPING: bool>(bencher: Bencher, nchunks: usize) {
    let step = if OVERLAPPING { 1 } else { 4 };
    let elements = PrimitiveArray::from_iter(0..(step * ROWS + 4) as u64).into_array();
    let offsets: Buffer<u64> = (0..ROWS).map(|idx| (step * idx) as u64).collect();
    let sizes: Buffer<u64> = std::iter::repeat_n(4, ROWS).collect();
    let chunk = ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    );
    // SAFETY: with step 4, every view ends exactly where the next begins.
    let chunk = unsafe { chunk.with_zero_copy_to_list(!OVERLAPPING) };
    canonicalize(bencher, chunk.into_array(), nchunks);
}
