// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// Benchmarks zip on VarBinView arrays with a highly fragmented mask (worst case for per-slice lookup paths).
#[divan::bench]
fn varbinview_zip_fragmented_mask(bencher: Bencher) {
    let len = 65_536;
    let if_true = fixture(len).into_array();
    let if_false = other_fixture(len).into_array();
    let mask = alternating_mask(len);

    bencher
        .with_inputs(|| {
            (
                if_true.clone(),
                if_false.clone(),
                mask.clone().into_array(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(t, f, m, ctx)| {
            m.zip(t.clone(), f.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        });
}

/// Benchmarks zip on VarBinView arrays with blocky mask segments to contrast with the fragmented case.
#[divan::bench]
fn varbinview_zip_block_mask(bencher: Bencher) {
    let len = 65_536;
    let if_true = fixture(len).into_array();
    let if_false = other_fixture(len).into_array();
    let mask = block_mask(len, 128);

    bencher
        .with_inputs(|| {
            (
                if_true.clone(),
                if_false.clone(),
                mask.clone().into_array(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(t, f, m, ctx)| {
            m.zip(t.clone(), f.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        });
}

fn fixture(len: usize) -> VarBinViewArray {
    VarBinViewArray::from_iter(
        [
            Some("short"),
            None,
            Some("this is a much longer string to force outlining"),
        ]
        .into_iter()
        .cycle()
        .take(len),
        DType::Utf8(Nullability::Nullable),
    )
}

fn other_fixture(len: usize) -> VarBinViewArray {
    VarBinViewArray::from_iter(
        [
            Some("different"),
            Some("another longer string that will be outlined as well"),
            None,
        ]
        .into_iter()
        .cycle()
        .take(len),
        DType::Utf8(Nullability::Nullable),
    )
}

fn alternating_mask(len: usize) -> Mask {
    Mask::from_iter((0..len).map(|i| i % 2 == 0))
}

fn block_mask(len: usize, block: usize) -> Mask {
    Mask::from_iter((0..len).map(|i| (i / block).is_multiple_of(2)))
}
