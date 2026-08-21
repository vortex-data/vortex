// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`super::FilterArray`].
//!
//! The main entrypoint is [`execute_filter`] which filters any [`Canonical`] array.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Filter;
use crate::arrays::Map;
use crate::arrays::MapArray;
use crate::arrays::NullArray;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::filter::FilterReduce;
use crate::arrays::fixed_width;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::scalar::Scalar;
use crate::validity::Validity;

mod bitbuffer;
mod bool;
pub(crate) mod buffer;
pub(crate) mod byte_compress;
mod fixed_size_list;
mod listview;
mod slice;
mod struct_;
mod take;
mod union;
mod varbinview;

/// Lazily filters a [`Validity`] with a partially selective mask.
pub(crate) fn filter_validity(validity: Validity, mask: &MaskValuesRef) -> Validity {
    validity
        .filter(&Mask::Values(MaskValuesRef::clone(mask)))
        .vortex_expect("filtering validity with a partially selective mask is valid")
}

/// Returns an all-null result when the child contains no valid values.
pub(super) fn execute_all_null_filter_fast_path(
    array: ArrayView<'_, Filter>,
    selected_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let child = array.array();
    if child
        .validity()?
        .execute_mask(child.len(), ctx)?
        .true_count()
        == 0
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), selected_count).into_array(),
        ));
    }

    Ok(None)
}

/// Filter a canonical array by a mask, returning a new canonical array.
pub(super) fn execute_filter(canonical: Canonical, mask: &MaskValuesRef) -> Canonical {
    match canonical {
        Canonical::Null(_) => Canonical::Null(NullArray::new(mask.true_count())),
        Canonical::Bool(a) => Canonical::Bool(bool::filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(fixed_width::filter::filter(&a, mask)),
        Canonical::Decimal(a) => Canonical::Decimal(fixed_width::filter::filter(&a, mask)),
        Canonical::VarBinView(a) => Canonical::VarBinView(varbinview::filter_varbinview(&a, mask)),
        Canonical::List(a) => Canonical::List(listview::filter_listview(&a, mask)),
        Canonical::Map(a) => Canonical::Map(filter_map(&a, mask)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(fixed_size_list::filter_fixed_size_list(&a, mask))
        }
        Canonical::Struct(a) => Canonical::Struct(struct_::filter_struct(&a, mask)),
        Canonical::Union(a) => Canonical::Union(union::filter_union(&a, mask)),
        Canonical::Extension(a) => {
            let filtered_storage = a
                .storage_array()
                .filter(Mask::Values(MaskValuesRef::clone(mask)))
                .vortex_expect("ExtensionArray storage type somehow could not be filtered");
            Canonical::Extension(ExtensionArray::new(a.ext_dtype().clone(), filtered_storage))
        }
        Canonical::Variant(a) => {
            let filter_mask = Mask::Values(MaskValuesRef::clone(mask));
            let filtered_core_storage = a
                .core_storage()
                .filter(filter_mask.clone())
                .vortex_expect("VariantArray core_storage could not be filtered");
            let filtered_shredded = a.shredded().map(|shredded| {
                shredded
                    .filter(filter_mask)
                    .vortex_expect("VariantArray shredded child could not be filtered")
            });
            Canonical::Variant(
                VariantArray::try_new(filtered_core_storage, filtered_shredded)
                    .vortex_expect("filtered VariantArray children are row-aligned"),
            )
        }
    }
}

fn filter_map(array: &MapArray, mask: &MaskValuesRef) -> MapArray {
    let filter_mask = Mask::Values(MaskValuesRef::clone(mask));
    let filtered = <Map as FilterReduce>::filter(array.as_view(), &filter_mask)
        .vortex_expect("MapArray somehow could not be filtered")
        .vortex_expect("Map filter reduce always produces an array");
    filtered.as_::<Map>().into_owned()
}
