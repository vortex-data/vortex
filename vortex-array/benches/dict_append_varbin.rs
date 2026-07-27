// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks appending a UTF-8 `DictArray` into a `DynVarBinBuilder`, the path a `pa.string()`
//! export of a low-cardinality string column takes.
//!
//! `cardinality` is the dictionary size; the win grows as it falls relative to `len`, because the
//! skipped intermediate is proportional to `len`.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::DynVarBinBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const SHAPES: &[(usize, usize)] = &[(4096, 16), (4096, 256), (4096, 2048)];

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn dict(len: usize, cardinality: usize, nulls: bool) -> DictArray {
    let values = VarBinViewArray::from_iter(
        (0..cardinality)
            .map(|i| Some(format!("https://example.com/some/path/segment/{i}").into_bytes())),
        DType::Utf8(Nullability::Nullable),
    );
    let codes = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if nulls && i.is_multiple_of(4) {
            None
        } else {
            Some(u32::try_from(i % cardinality).unwrap())
        }
    }));
    DictArray::try_new(codes.into_array(), values.into_array()).unwrap()
}

fn bench_append(bencher: Bencher, array: DictArray) {
    let mut ctx = SESSION.create_execution_ctx();
    let len = array.len();
    let dtype = array.dtype().clone();
    let array = array.into_array();
    bencher.bench_local(|| {
        let mut builder = DynVarBinBuilder::with_capacity(dtype.clone(), false, len);
        array.append_to_builder(&mut builder, &mut ctx).unwrap();
        black_box(builder.finish_into_varbin())
    });
}

#[divan::bench(args = SHAPES)]
fn all_valid(bencher: Bencher, shape: (usize, usize)) {
    bench_append(bencher, dict(shape.0, shape.1, false));
}

#[divan::bench(args = SHAPES)]
fn with_null_codes(bencher: Bencher, shape: (usize, usize)) {
    bench_append(bencher, dict(shape.0, shape.1, true));
}
