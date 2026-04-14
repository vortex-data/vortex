// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`FilterArray`].
//!
//! The main entrypoint is [`execute_filter`] which filters any [`Canonical`] array.

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Filter;
use crate::arrays::NullArray;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::variant::VariantArrayExt;
use crate::scalar::Scalar;
use crate::validity::Validity;

mod bitbuffer;
mod bool;
mod buffer;
mod decimal;
mod fixed_size_list;
mod listview;
mod primitive;
mod slice;
mod struct_;
mod varbinview;

/// Reconstruct a [`Mask`] from an [`Arc<MaskValues>`].
fn values_to_mask(values: &Arc<MaskValues>) -> Mask {
    Mask::Values(Arc::clone(values))
}

/// A helper function that lazily filters a [`Validity`] with selection mask values.
fn filter_validity(validity: Validity, mask: &Arc<MaskValues>) -> Validity {
    validity
        .filter(&values_to_mask(mask))
        .vortex_expect("Somehow unable to wrap filter around a validity array")
}

/// Check for some fast-path execution conditions before calling [`execute_filter`].
pub(super) fn execute_filter_fast_paths(
    array: ArrayView<'_, Filter>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let true_count = array.mask.true_count();

    // If the mask selects nothing, the output is empty.
    if true_count == 0 {
        return Ok(Some(Canonical::empty(array.dtype()).into_array()));
    }

    // If the mask selects everything, then we can just fully decompress the whole thing.
    if true_count == array.mask.len() {
        return Ok(Some(array.child().clone()));
    }

    // Also check if the array itself is completely null, in which case we only care about the total
    // number of nulls, not the values.
    let child_arr = array.array();
    if child_arr
        .validity()?
        .to_mask(child_arr.len(), ctx)?
        .true_count()
        == 0
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), true_count).into_array(),
        ));
    }

    Ok(None)
}

/// Filter a canonical array by a mask, returning a new canonical array.
pub(super) fn execute_filter(canonical: Canonical, mask: &Arc<MaskValues>) -> Canonical {
    match canonical {
        Canonical::Null(_) => Canonical::Null(NullArray::new(mask.true_count())),
        Canonical::Bool(a) => Canonical::Bool(bool::filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(primitive::filter_primitive(&a, mask)),
        Canonical::Decimal(a) => Canonical::Decimal(decimal::filter_decimal(&a, mask)),
        Canonical::VarBinView(a) => Canonical::VarBinView(varbinview::filter_varbinview(&a, mask)),
        Canonical::List(a) => Canonical::List(listview::filter_listview(&a, mask)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(fixed_size_list::filter_fixed_size_list(&a, mask))
        }
        Canonical::Struct(a) => Canonical::Struct(struct_::filter_struct(&a, mask)),
        Canonical::Extension(a) => {
            let filtered_storage = a
                .storage_array()
                .filter(values_to_mask(mask))
                .vortex_expect("ExtensionArray storage type somehow could not be filtered");
            Canonical::Extension(ExtensionArray::new(a.ext_dtype().clone(), filtered_storage))
        }
        Canonical::Variant(a) => {
            let filtered_child = a
                .child()
                .filter(values_to_mask(mask))
                .vortex_expect("VariantArray child could not be filtered");
            Canonical::Variant(VariantArray::new(filtered_child))
        }
    }
}
