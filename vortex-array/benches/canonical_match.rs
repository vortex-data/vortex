// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare canonical classification with the original exhaustive downcast chain.

use std::hint::black_box;

use divan::Bencher;
use vortex_array::AnyCanonical;
use vortex_array::ArrayRef;
use vortex_array::CanonicalView;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Map;
use vortex_array::arrays::Null;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::Union;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::Variant;
use vortex_buffer::buffer;

fn main() {
    divan::main();
}

const CASES: &[&str] = &["null", "primitive", "varbinview", "constant"];

fn array(case: &str) -> ArrayRef {
    match case {
        "null" => NullArray::new(4).into_array(),
        "primitive" => buffer![1i32, 2, 3, 4].into_array(),
        "varbinview" => VarBinViewArray::from_iter_str(["a", "b", "c", "d"]).into_array(),
        "constant" => ConstantArray::new(1i32, 4).into_array(),
        _ => unreachable!(),
    }
}

#[divan::bench(args = CASES)]
fn matches_vtable(bencher: Bencher, case: &str) {
    let array = array(case);
    bencher.bench_local(|| black_box(black_box(&array).is::<AnyCanonical>()));
}

#[divan::bench(args = CASES)]
fn matches_exhaustive(bencher: Bencher, case: &str) {
    let array = array(case);
    bencher.bench_local(|| black_box(matches_exhaustively(black_box(&array))));
}

#[divan::bench(args = CASES)]
fn try_match_vtable(bencher: Bencher, case: &str) {
    let array = array(case);
    bencher.bench_local(|| black_box(black_box(&array).as_opt::<AnyCanonical>()));
}

#[divan::bench(args = CASES)]
fn try_match_exhaustive(bencher: Bencher, case: &str) {
    let array = array(case);
    bencher.bench_local(|| black_box(try_match_exhaustively(black_box(&array))));
}

#[inline]
fn matches_exhaustively(array: &ArrayRef) -> bool {
    array.is::<Null>()
        || array.is::<Bool>()
        || array.is::<Primitive>()
        || array.is::<Decimal>()
        || array.is::<Struct>()
        || array.is::<Union>()
        || array.is::<ListView>()
        || array.is::<Map>()
        || array.is::<FixedSizeList>()
        || array.is::<VarBinView>()
        || array.is::<Variant>()
        || array.is::<Extension>()
}

#[inline]
fn try_match_exhaustively(array: &ArrayRef) -> Option<CanonicalView<'_>> {
    if let Some(a) = array.as_opt::<Null>() {
        Some(CanonicalView::Null(a))
    } else if let Some(a) = array.as_opt::<Bool>() {
        Some(CanonicalView::Bool(a))
    } else if let Some(a) = array.as_opt::<Primitive>() {
        Some(CanonicalView::Primitive(a))
    } else if let Some(a) = array.as_opt::<Decimal>() {
        Some(CanonicalView::Decimal(a))
    } else if let Some(a) = array.as_opt::<Struct>() {
        Some(CanonicalView::Struct(a))
    } else if let Some(a) = array.as_opt::<Union>() {
        Some(CanonicalView::Union(a))
    } else if let Some(a) = array.as_opt::<ListView>() {
        Some(CanonicalView::List(a))
    } else if let Some(a) = array.as_opt::<Map>() {
        Some(CanonicalView::Map(a))
    } else if let Some(a) = array.as_opt::<FixedSizeList>() {
        Some(CanonicalView::FixedSizeList(a))
    } else if let Some(a) = array.as_opt::<VarBinView>() {
        Some(CanonicalView::VarBinView(a))
    } else if let Some(a) = array.as_opt::<Variant>() {
        Some(CanonicalView::Variant(a))
    } else {
        array.as_opt::<Extension>().map(CanonicalView::Extension)
    }
}
