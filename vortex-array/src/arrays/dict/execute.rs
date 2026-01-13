// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for DictArray - takes from values using codes (indices).

use vortex_error::VortexExpect;

use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::TakeKernel;

/// TODO: replace usage of compute fn.
/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
pub fn take_canonical(values: Canonical, codes: &PrimitiveArray) -> Canonical {
    match values {
        Canonical::Null(a) => Canonical::Null(take_null(&a, codes)),
        Canonical::Bool(a) => Canonical::Bool(take_bool(&a, codes)),
        Canonical::Primitive(a) => Canonical::Primitive(take_primitive(&a, codes)),
        Canonical::Decimal(a) => Canonical::Decimal(take_decimal(&a, codes)),
        Canonical::VarBinView(a) => Canonical::VarBinView(take_varbinview(&a, codes)),
        Canonical::List(a) => Canonical::List(take_listview(&a, codes)),
        Canonical::FixedSizeList(a) => Canonical::FixedSizeList(take_fixed_size_list(&a, codes)),
        Canonical::Struct(a) => Canonical::Struct(take_struct(&a, codes)),
        Canonical::Extension(a) => Canonical::Extension(take_extension(&a, codes)),
    }
}

fn take_null(_array: &NullArray, codes: &PrimitiveArray) -> NullArray {
    NullVTable
        .take(_array, codes.as_ref())
        .vortex_expect("take null array")
        .as_::<NullVTable>()
        .clone()
}

fn take_bool(array: &BoolArray, codes: &PrimitiveArray) -> BoolArray {
    BoolVTable
        .take(array, codes.as_ref())
        .vortex_expect("take bool array")
        .as_::<BoolVTable>()
        .clone()
}

fn take_primitive(array: &PrimitiveArray, codes: &PrimitiveArray) -> PrimitiveArray {
    PrimitiveVTable
        .take(array, codes.as_ref())
        .vortex_expect("take primitive array")
        .as_::<PrimitiveVTable>()
        .clone()
}

fn take_decimal(array: &DecimalArray, codes: &PrimitiveArray) -> DecimalArray {
    DecimalVTable
        .take(array, codes.as_ref())
        .vortex_expect("take decimal array")
        .as_::<DecimalVTable>()
        .clone()
}

fn take_varbinview(array: &VarBinViewArray, codes: &PrimitiveArray) -> VarBinViewArray {
    VarBinViewVTable
        .take(array, codes.as_ref())
        .vortex_expect("take varbinview array")
        .as_::<VarBinViewVTable>()
        .clone()
}

fn take_listview(array: &ListViewArray, codes: &PrimitiveArray) -> ListViewArray {
    ListViewVTable
        .take(array, codes.as_ref())
        .vortex_expect("take listview array")
        .as_::<ListViewVTable>()
        .clone()
}

fn take_fixed_size_list(array: &FixedSizeListArray, codes: &PrimitiveArray) -> FixedSizeListArray {
    FixedSizeListVTable
        .take(array, codes.as_ref())
        .vortex_expect("take fixed size list array")
        .as_::<FixedSizeListVTable>()
        .clone()
}

fn take_struct(array: &StructArray, codes: &PrimitiveArray) -> StructArray {
    StructVTable
        .take(array, codes.as_ref())
        .vortex_expect("take struct array")
        .as_::<StructVTable>()
        .clone()
}

fn take_extension(array: &ExtensionArray, codes: &PrimitiveArray) -> ExtensionArray {
    use crate::compute::take;

    let taken_storage =
        take(array.storage(), codes.as_ref()).vortex_expect("take extension storage");
    ExtensionArray::new(array.ext_dtype().clone(), taken_storage)
}
