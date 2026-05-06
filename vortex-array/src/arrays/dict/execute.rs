// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for DictArray - takes from values using codes (indices).

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::CanonicalView;
use crate::ExecutionCtx;
use crate::IntoArray;
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
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::VariantArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::variant::VariantArrayExt;

/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
pub(crate) fn take_canonical(
    values: CanonicalView,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    let values = Canonical::from(values);
    Ok(match values {
        Canonical::Null(a) => Canonical::Null(take_null(&a, codes)),
        Canonical::Bool(a) => Canonical::Bool(take_bool(&a, codes, ctx)?),
        Canonical::Primitive(a) => Canonical::Primitive(take_primitive(&a, codes, ctx)),
        Canonical::Decimal(a) => Canonical::Decimal(take_decimal(&a, codes, ctx)),
        Canonical::VarBinView(a) => Canonical::VarBinView(take_varbinview(&a, codes, ctx)),
        Canonical::List(a) => Canonical::List(take_listview(&a, codes, ctx)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(take_fixed_size_list(&a, codes, ctx))
        }
        Canonical::Struct(a) => Canonical::Struct(take_struct(&a, codes)),
        Canonical::Extension(a) => Canonical::Extension(take_extension(&a, codes, ctx)),
        Canonical::Variant(a) => {
            let taken_child = a
                .child()
                .take(codes.clone().into_array())
                .vortex_expect("VariantArray child could not be taken");
            Canonical::Variant(VariantArray::new(taken_child))
        }
    })
}

/// Take for NullArray is trivial - just create a new NullArray with the new length.
fn take_null(_array: &NullArray, codes: &PrimitiveArray) -> NullArray {
    NullArray::new(codes.len())
}

// TODO(joe): use dict_bool_take
fn take_bool(
    array: &BoolArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BoolArray> {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    Ok(<Bool as TakeExecute>::take(array, &codes_ref, ctx)?
        .vortex_expect("take bool should not return None")
        .as_::<Bool>()
        .into_owned())
}

fn take_primitive(
    array: &PrimitiveArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> PrimitiveArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <Primitive as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take primitive array")
        .vortex_expect("take primitive should not return None")
        .as_::<Primitive>()
        .into_owned()
}

fn take_decimal(
    array: &DecimalArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> DecimalArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <Decimal as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take decimal array")
        .vortex_expect("take decimal should not return None")
        .as_::<Decimal>()
        .into_owned()
}

fn take_varbinview(
    array: &VarBinViewArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VarBinViewArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <VarBinView as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take varbinview array")
        .vortex_expect("take varbinview should not return None")
        .as_::<VarBinView>()
        .into_owned()
}

fn take_listview(
    array: &ListViewArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ListViewArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <ListView as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take listview execute")
        .vortex_expect("ListView TakeExecute should not return None")
        .as_::<ListView>()
        .into_owned()
}

fn take_fixed_size_list(
    array: &FixedSizeListArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <FixedSizeList as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take fixed size list array")
        .vortex_expect("take fixed size list should not return None")
        .as_::<FixedSizeList>()
        .into_owned()
}

fn take_struct(array: &StructArray, codes: &PrimitiveArray) -> StructArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <Struct as TakeReduce>::take(array, &codes_ref)
        .vortex_expect("take struct array")
        .vortex_expect("take struct should not return None")
        .as_::<Struct>()
        .into_owned()
}

fn take_extension(
    array: &ExtensionArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ExtensionArray {
    let codes_ref = codes.clone().into_array();
    let array = array.as_view();
    <Extension as TakeExecute>::take(array, &codes_ref, ctx)
        .vortex_expect("take extension storage")
        .vortex_expect("take extension should not return None")
        .as_::<Extension>()
        .into_owned()
}
