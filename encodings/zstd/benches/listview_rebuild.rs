// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::{ListViewArray, ListViewRebuildMode, VarBinViewArray};
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_zstd::ZstdArray;

#[divan::bench]
fn rebuild_naive(bencher: Bencher) {
    let dudes = VarBinViewArray::from_iter_str(["Washington", "Adams", "Jefferson", "Madison"])
        .into_array();
    let dudes = ZstdArray::from_array(dudes, 9, 1024).unwrap().into_array();

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

    bencher.bench_local(|| list_view.rebuild(ListViewRebuildMode::MakeZeroCopyToList))
}

fn main() {
    divan::main()
}
