// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Array;
use vortex_array::arrays::VarBinViewArray;
use vortex_fsst::{FSSTEncoding, FSSTViewEncoding};

#[divan::bench]
fn fsst(bencher: Bencher) {
    let canonical = VarBinViewArray::from_iter_nullable_str(
        [
            None,
            None,
            Some("abcdefghijklmnopqrstuvwxyz"),
            Some("short"),
            None,
            Some("abcdfghijklmnstuvwxyz"),
        ]
        .into_iter()
        .cycle()
        .take(65_536),
    );

    let compressed = FSSTEncoding
        .encode(&canonical.to_canonical(), None)
        .unwrap()
        .unwrap();

    bencher.bench_local(|| compressed.to_canonical())
}

// Canonicalize the FSST View array
#[divan::bench]
fn fsst_view(bencher: Bencher) {
    let canonical = VarBinViewArray::from_iter_nullable_str(
        [
            None,
            None,
            Some("abcdefghijklmnopqrstuvwxyz"),
            Some("short"),
            None,
            Some("abcdfghijklmnstuvwxyz"),
        ]
        .into_iter()
        .cycle()
        .take(65_535),
    );

    let compressed = FSSTViewEncoding
        .encode(&canonical.to_canonical(), None)
        .unwrap()
        .unwrap();

    bencher.bench_local(|| compressed.to_canonical())
}

fn main() {
    divan::main()
}
