// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for DictArray - takes from values using codes (indices).

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayView;
use crate::Canonical;
use crate::CanonicalView;
use crate::ExecutionCtx;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Map;
use crate::arrays::MapArray;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::VariantArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::variant::VariantArraySlotsExt;

/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
pub(crate) fn take_canonical(
    values: CanonicalView,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    Ok(match values {
        CanonicalView::Null(a) => Canonical::Null(take_null(a, codes)),
        CanonicalView::Bool(a) => Canonical::Bool(take_bool(a, codes, ctx)?),
        CanonicalView::Primitive(a) => Canonical::Primitive(take_primitive(a, codes, ctx)),
        CanonicalView::Decimal(a) => Canonical::Decimal(take_decimal(a, codes, ctx)),
        CanonicalView::VarBinView(a) => Canonical::VarBinView(take_varbinview(a, codes, ctx)),
        CanonicalView::List(a) => Canonical::List(take_listview(a, codes, ctx)),
        CanonicalView::Map(a) => Canonical::Map(take_map(a, codes, ctx)),
        CanonicalView::FixedSizeList(a) => {
            Canonical::FixedSizeList(take_fixed_size_list(a, codes, ctx))
        }
        CanonicalView::Struct(a) => Canonical::Struct(take_struct(a, codes)),
        CanonicalView::Union(a) => Canonical::Union(take_union(a, codes)),
        CanonicalView::Extension(a) => Canonical::Extension(take_extension(a, codes, ctx)),
        CanonicalView::Variant(a) => {
            let indices = codes.array().clone();
            let taken_core_storage = a.core_storage().take(indices.clone())?;
            let taken_shredded = a
                .shredded()
                .map(|shredded| shredded.take(indices))
                .transpose()?;
            Canonical::Variant(VariantArray::try_new(taken_core_storage, taken_shredded)?)
        }
    })
}

/// Take for NullArray is trivial - just create a new NullArray with the new length.
fn take_null(_array: ArrayView<'_, Null>, codes: ArrayView<'_, Primitive>) -> NullArray {
    NullArray::new(codes.len())
}

fn take_bool(
    array: ArrayView<'_, Bool>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BoolArray> {
    let codes_ref = codes.array();
    Ok(<Bool as TakeExecute>::take(array, codes_ref, ctx)?
        .vortex_expect("take bool should not return None")
        .as_::<Bool>()
        .into_owned())
}

fn take_primitive(
    array: ArrayView<'_, Primitive>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> PrimitiveArray {
    let codes_ref = codes.array();
    <Primitive as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take primitive array")
        .vortex_expect("take primitive should not return None")
        .as_::<Primitive>()
        .into_owned()
}

fn take_decimal(
    array: ArrayView<'_, Decimal>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> DecimalArray {
    let codes_ref = codes.array();
    <Decimal as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take decimal array")
        .vortex_expect("take decimal should not return None")
        .as_::<Decimal>()
        .into_owned()
}

fn take_varbinview(
    array: ArrayView<'_, VarBinView>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VarBinViewArray {
    let codes_ref = codes.array();
    <VarBinView as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take varbinview array")
        .vortex_expect("take varbinview should not return None")
        .as_::<VarBinView>()
        .into_owned()
}

fn take_listview(
    array: ArrayView<'_, ListView>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> ListViewArray {
    let codes_ref = codes.array();
    <ListView as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take listview execute")
        .vortex_expect("ListView TakeExecute should not return None")
        .as_::<ListView>()
        .into_owned()
}

fn take_map(
    array: ArrayView<'_, Map>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> MapArray {
    <Map as TakeExecute>::take(array, codes.array(), ctx)
        .vortex_expect("take map execute")
        .vortex_expect("Map TakeExecute should not return None")
        .as_::<Map>()
        .into_owned()
}

fn take_fixed_size_list(
    array: ArrayView<'_, FixedSizeList>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    let codes_ref = codes.array();
    <FixedSizeList as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take fixed size list array")
        .vortex_expect("take fixed size list should not return None")
        .as_::<FixedSizeList>()
        .into_owned()
}

fn take_struct(array: ArrayView<'_, Struct>, codes: ArrayView<'_, Primitive>) -> StructArray {
    let codes_ref = codes.array();
    <Struct as TakeReduce>::take(array, codes_ref)
        .vortex_expect("take struct array")
        .vortex_expect("take struct should not return None")
        .as_::<Struct>()
        .into_owned()
}

fn take_union(array: ArrayView<'_, Union>, codes: ArrayView<'_, Primitive>) -> UnionArray {
    let codes_ref = codes.array();
    <Union as TakeReduce>::take(array, codes_ref)
        .vortex_expect("take union array")
        .vortex_expect("take union should not return None")
        .as_::<Union>()
        .into_owned()
}

fn take_extension(
    array: ArrayView<'_, Extension>,
    codes: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> ExtensionArray {
    let codes_ref = codes.array();
    <Extension as TakeExecute>::take(array, codes_ref, ctx)
        .vortex_expect("take extension storage")
        .vortex_expect("take extension should not return None")
        .as_::<Extension>()
        .into_owned()
}
