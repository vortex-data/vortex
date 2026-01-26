// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`FilterArray`].
//!
//! The main entrypoint is [`execute_filter`] which filters any [`Canonical`] array.

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewRebuildMode;
use crate::arrays::ListViewVTable;
use crate::arrays::NullArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::FilterKernel;

mod primitive;

// TODO(connor): Stop using the old compute kernels and move all code into this module.
/// Filter a canonical array by a mask, returning a new canonical array.
pub(super) fn execute_filter(canonical: Canonical, mask: &Mask) -> Canonical {
    match canonical {
        Canonical::Null(a) => Canonical::Null(filter_null(&a, mask)),
        Canonical::Bool(a) => Canonical::Bool(filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(primitive::filter_primitive(&a, mask)),
        Canonical::Decimal(a) => Canonical::Decimal(filter_decimal(&a, mask)),
        Canonical::VarBinView(a) => Canonical::VarBinView(filter_varbinview(&a, mask)),
        Canonical::List(a) => Canonical::List(filter_listview(&a, mask)),
        Canonical::FixedSizeList(a) => Canonical::FixedSizeList(filter_fixed_size_list(&a, mask)),
        Canonical::Struct(a) => Canonical::Struct(filter_struct(&a, mask)),
        Canonical::Extension(a) => Canonical::Extension(filter_extension(&a, mask)),
    }
}

fn filter_null(_array: &NullArray, mask: &Mask) -> NullArray {
    NullArray::new(mask.true_count())
}

fn filter_bool(array: &BoolArray, mask: &Mask) -> BoolArray {
    BoolVTable
        .filter(array, mask)
        .vortex_expect("filter bool array")
        .as_::<BoolVTable>()
        .clone()
}

fn filter_decimal(array: &DecimalArray, mask: &Mask) -> DecimalArray {
    DecimalVTable
        .filter(array, mask)
        .vortex_expect("filter decimal array")
        .as_::<DecimalVTable>()
        .clone()
}

/// Filter a VarBinViewArray - delegates to Arrow filter.
fn filter_varbinview(array: &VarBinViewArray, mask: &Mask) -> VarBinViewArray {
    VarBinViewVTable
        .filter(array, mask)
        .vortex_expect("filter varbinview array")
        .as_::<VarBinViewVTable>()
        .clone()
}

fn filter_listview(array: &ListViewArray, mask: &Mask) -> ListViewArray {
    ListViewVTable
        .filter(array, mask)
        .vortex_expect("filter listview array")
        .as_::<ListViewVTable>()
        .clone()
        .rebuild(ListViewRebuildMode::MakeZeroCopyToList)
        .vortex_expect("rebuild listview array")
}

fn filter_fixed_size_list(array: &FixedSizeListArray, mask: &Mask) -> FixedSizeListArray {
    FixedSizeListVTable
        .filter(array, mask)
        .vortex_expect("filter fixed size list array")
        .as_::<FixedSizeListVTable>()
        .clone()
}

fn filter_struct(array: &StructArray, mask: &Mask) -> StructArray {
    StructVTable
        .filter(array, mask)
        .vortex_expect("filter struct array")
        .as_::<StructVTable>()
        .clone()
}

fn filter_extension(array: &ExtensionArray, mask: &Mask) -> ExtensionArray {
    let filtered_storage = array
        .storage()
        .filter(mask.clone())
        .vortex_expect("filter extension storage");
    ExtensionArray::new(array.ext_dtype().clone(), filtered_storage)
}
