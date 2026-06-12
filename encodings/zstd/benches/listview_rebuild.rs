// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;
use vortex_zstd::ZstdData;

/// A shared session for the `ListView` rebuild benchmark, used to create execution contexts.
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench(sample_size = 1000)]
fn rebuild_naive(bencher: Bencher) {
    let dudes = VarBinViewArray::from_iter_str(["Washington", "Adams", "Jefferson", "Madison"])
        .into_array();
    let dtype = dudes.dtype().clone();
    let validity = dudes.validity().unwrap();
    let dudes = Zstd::try_new(
        dtype,
        ZstdData::from_array(dudes, 9, 1024, &mut SESSION.create_execution_ctx()).unwrap(),
        validity,
    )
    .unwrap()
    .into_array();

    let offsets = std::iter::repeat_n(0u32, 1024)
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = [0u64, 1, 2, 3, 4]
        .into_iter()
        .cycle()
        .take(1024)
        .collect::<Buffer<u64>>()
        .into_array();

    let list_view = ListViewArray::new(dudes, offsets, sizes, Validity::NonNullable);

    bencher
        .with_inputs(|| (&list_view, SESSION.create_execution_ctx()))
        .bench_refs(|(list_view, ctx)| {
            list_view.rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)
        })
}

fn main() {
    divan::main()
}
