// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for DictArray - takes from values using codes (indices).

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::TakeExecute;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;

/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
pub fn take_canonical(
    values: Canonical,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
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
        Canonical::Struct(a) => Canonical::Struct(take_struct(&a, codes, ctx)),
        Canonical::Extension(a) => Canonical::Extension(take_extension(&a, codes, ctx)),
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
    Ok(
        <BoolVTable as TakeExecute>::take(array, &codes.to_array(), ctx)?
            .vortex_expect("take bool should not return None")
            .as_::<BoolVTable>()
            .clone(),
    )
}

fn take_primitive(
    array: &PrimitiveArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> PrimitiveArray {
    <PrimitiveVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take primitive array")
        .vortex_expect("take primitive should not return None")
        .as_::<PrimitiveVTable>()
        .clone()
}

fn take_decimal(
    array: &DecimalArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> DecimalArray {
    <DecimalVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take decimal array")
        .vortex_expect("take decimal should not return None")
        .as_::<DecimalVTable>()
        .clone()
}

fn take_varbinview(
    array: &VarBinViewArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VarBinViewArray {
    <VarBinViewVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take varbinview array")
        .vortex_expect("take varbinview should not return None")
        .as_::<VarBinViewVTable>()
        .clone()
}

fn take_listview(
    array: &ListViewArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ListViewArray {
    <ListViewVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take listview array")
        .vortex_expect("take listview should not return None")
        .as_::<ListViewVTable>()
        .clone()
}

fn take_fixed_size_list(
    array: &FixedSizeListArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    <FixedSizeListVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take fixed size list array")
        .vortex_expect("take fixed size list should not return None")
        .as_::<FixedSizeListVTable>()
        .clone()
}

fn take_struct(array: &StructArray, codes: &PrimitiveArray, ctx: &mut ExecutionCtx) -> StructArray {
    <StructVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take struct array")
        .vortex_expect("take struct should not return None")
        .as_::<StructVTable>()
        .clone()
}

fn take_extension(
    array: &ExtensionArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ExtensionArray {
    <ExtensionVTable as TakeExecute>::take(array, &codes.to_array(), ctx)
        .vortex_expect("take extension storage")
        .vortex_expect("take extension should not return None")
        .as_::<ExtensionVTable>()
        .clone()
}
