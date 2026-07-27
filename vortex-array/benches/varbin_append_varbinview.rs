// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks appending a `VarBinViewArray` into a `DynVarBinBuilder`, the fallback path every
//! string encoding without a direct branch reaches through canonicalization.
//!
//! `inlined` values are 12 bytes or fewer and live in the view itself, which is the common case for
//! short strings; `heap` values exceed that and live in a data buffer.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::DynVarBinBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4096, 16384];

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn inlined_values(len: usize, nulls: bool) -> VarBinViewArray {
    VarBinViewArray::from_iter(
        (0..len).map(|i| {
            (!nulls || !i.is_multiple_of(4)).then(|| format!("{:<12}", i % 1000).into_bytes())
        }),
        DType::Utf8(Nullability::Nullable),
    )
}

fn heap_values(len: usize, nulls: bool) -> VarBinViewArray {
    VarBinViewArray::from_iter(
        (0..len).map(|i| {
            (!nulls || !i.is_multiple_of(4))
                .then(|| format!("https://example.com/some/path/segment/{i}").into_bytes())
        }),
        DType::Utf8(Nullability::Nullable),
    )
}

fn bench_append(bencher: Bencher, array: VarBinViewArray) {
    let mut ctx = SESSION.create_execution_ctx();
    let len = array.len();
    let array = array.into_array();
    bencher.bench_local(|| {
        let mut builder =
            DynVarBinBuilder::with_capacity(DType::Utf8(Nullability::Nullable), false, len);
        array.append_to_builder(&mut builder, &mut ctx).unwrap();
        black_box(builder.finish_into_varbin())
    });
}

#[divan::bench(args = SIZES)]
fn inlined_all_valid(bencher: Bencher, len: usize) {
    bench_append(bencher, inlined_values(len, false));
}

#[divan::bench(args = SIZES)]
fn inlined_with_nulls(bencher: Bencher, len: usize) {
    bench_append(bencher, inlined_values(len, true));
}

#[divan::bench(args = SIZES)]
fn heap_all_valid(bencher: Bencher, len: usize) {
    bench_append(bencher, heap_values(len, false));
}

#[divan::bench(args = SIZES)]
fn heap_with_nulls(bencher: Bencher, len: usize) {
    bench_append(bencher, heap_values(len, true));
}
