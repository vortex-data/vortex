// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for FilterArray - filters canonical arrays by a mask.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::Canonical;
use crate::IntoArray;
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
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::filter::FilterArray;
use crate::compute::FilterKernel;
use crate::compute::filter;
use crate::validity::Validity;

/// TODO: replace usage of compute fn.
/// Filter a canonical array by a mask, returning a new canonical array.
pub fn filter_canonical(canonical: Canonical, mask: &Mask) -> Canonical {
    match canonical {
        Canonical::Null(a) => Canonical::Null(filter_null(&a, mask)),
        Canonical::Bool(a) => Canonical::Bool(filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(filter_primitive(&a, mask)),
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

fn filter_primitive(array: &PrimitiveArray, mask: &Mask) -> PrimitiveArray {
    use vortex_dtype::match_each_native_ptype;

    // Lazy validity: wrap in FilterArray instead of eagerly filtering
    let validity = match array.validity().vortex_expect("primitive validity") {
        v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => v,
        Validity::Array(arr) => {
            Validity::Array(FilterArray::new(arr.clone(), mask.clone()).into_array())
        }
    };

    match_each_native_ptype!(array.ptype(), |T| {
        let filtered = filter_slice(
            array.as_slice::<T>(),
            mask,
            FILTER_SLICES_SELECTIVITY_THRESHOLD,
        );
        PrimitiveArray::new(filtered, validity)
    })
}

/// Threshold for choosing between indices vs slices filtering strategy.
pub const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

/// Filter a typed buffer by a mask, returning a new buffer with only the selected elements.
///
/// This is the core filtering operation used by both FilterArray execution and the
/// FilterKernel for primitive arrays.
///
/// # Arguments
/// * `values` - The source slice of values to filter
/// * `mask` - The mask indicating which elements to keep (must have `values()` available)
/// * `selectivity_threshold` - Threshold for choosing between indices vs slices strategy
pub fn filter_slice<T: Copy>(values: &[T], mask: &Mask, selectivity_threshold: f64) -> Buffer<T> {
    let mask_values = mask
        .values()
        .vortex_expect("AllTrue and AllFalse should be handled by caller");

    match mask_values.threshold_iter(selectivity_threshold) {
        MaskIter::Indices(indices) => indices
            .iter()
            .copied()
            .map(|idx| *unsafe { values.get_unchecked(idx) })
            .collect(),
        MaskIter::Slices(slices) => {
            let mut output = BufferMut::with_capacity(mask.true_count());
            for (start, end) in slices.iter().copied() {
                output.extend_from_slice(&values[start..end]);
            }
            output.freeze()
        }
    }
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
    let filtered_storage = filter(array.storage(), mask).vortex_expect("filter extension storage");
    ExtensionArray::new(array.ext_dtype().clone(), filtered_storage)
}
