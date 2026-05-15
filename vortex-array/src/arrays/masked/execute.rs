// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for MaskedArray - applies a validity mask to canonical arrays.

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::Canonical;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::VariantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

/// TODO: replace usage of compute fn.
/// Apply a validity mask to a canonical array, ANDing with existing validity.
///
/// This is the core operation for MaskedArray execution - it intersects the child's
/// validity with the provided mask, marking additional positions as invalid.
pub fn mask_validity_canonical(
    canonical: Canonical,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    Ok(match canonical {
        n @ Canonical::Null(_) => n,
        Canonical::Bool(a) => Canonical::Bool(mask_validity_bool(a, validity)?),
        Canonical::Primitive(a) => Canonical::Primitive(mask_validity_primitive(a, validity)?),
        Canonical::Decimal(a) => Canonical::Decimal(mask_validity_decimal(a, validity)?),
        Canonical::VarBinView(a) => Canonical::VarBinView(mask_validity_varbinview(a, validity)?),
        Canonical::List(a) => Canonical::List(mask_validity_listview(a, validity)?),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(mask_validity_fixed_size_list(a, validity)?)
        }
        Canonical::Struct(a) => Canonical::Struct(mask_validity_struct(a, validity)?),
        Canonical::Extension(a) => Canonical::Extension(mask_validity_extension(a, validity, ctx)?),
        Canonical::Variant(a) => Canonical::Variant(mask_validity_variant(a, validity, ctx)?),
    })
}

fn mask_validity_bool(array: BoolArray, mask: Validity) -> VortexResult<BoolArray> {
    let new_validity = Validity::and(array.validity()?, mask)?;
    Ok(BoolArray::new(array.to_bit_buffer(), new_validity))
}

fn mask_validity_primitive(
    array: PrimitiveArray,
    validity: Validity,
) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();
    let new_validity = Validity::and(array.validity()?, validity)?;
    // SAFETY: validity has same length as values
    Ok(unsafe {
        PrimitiveArray::new_unchecked_from_handle(
            array.buffer_handle().clone(),
            ptype,
            new_validity,
        )
    })
}

fn mask_validity_decimal(array: DecimalArray, validity: Validity) -> VortexResult<DecimalArray> {
    let new_validity = Validity::and(array.validity()?, validity)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        DecimalArray::new_unchecked_handle(
            array.buffer_handle().clone(),
            array.values_type(),
            array.decimal_dtype(),
            new_validity,
        )
    })
}

/// Mask validity for VarBinViewArray.
fn mask_validity_varbinview(
    array: VarBinViewArray,
    validity: Validity,
) -> VortexResult<VarBinViewArray> {
    let dtype = array.dtype().as_nullable();
    let new_validity = Validity::and(array.validity()?, validity)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(
            array.views_handle().clone(),
            Arc::clone(array.data_buffers()),
            dtype,
            new_validity,
        )
    })
}

fn mask_validity_listview(array: ListViewArray, validity: Validity) -> VortexResult<ListViewArray> {
    let new_validity = Validity::and(array.validity()?, validity)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        ListViewArray::new_unchecked(
            array.elements().clone(),
            array.offsets().clone(),
            array.sizes().clone(),
            new_validity,
        )
    })
}

fn mask_validity_fixed_size_list(
    array: FixedSizeListArray,
    validity: Validity,
) -> VortexResult<FixedSizeListArray> {
    let len = array.len();
    let list_size = array.list_size();
    let new_validity = Validity::and(array.validity()?, validity)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(array.elements().clone(), list_size, new_validity, len)
    })
}

fn mask_validity_struct(array: StructArray, validity: Validity) -> VortexResult<StructArray> {
    let len = array.len();
    let new_validity = Validity::and(array.validity()?, validity)?;
    let fields = array.unmasked_fields();
    let struct_fields = array.struct_fields();
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe { StructArray::new_unchecked(fields, struct_fields.clone(), len, new_validity) })
}

fn mask_validity_extension(
    array: ExtensionArray,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExtensionArray> {
    // For extension arrays, we need to mask the underlying storage
    let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
    let masked_storage = mask_validity_canonical(storage, validity, ctx)?;
    let masked_storage = masked_storage.into_array();
    Ok(ExtensionArray::new(
        array
            .ext_dtype()
            .with_nullability(masked_storage.dtype().nullability()),
        masked_storage,
    ))
}

fn mask_validity_variant(
    array: VariantArray,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VariantArray> {
    let core_storage = array.core_storage().clone();
    let len = core_storage.len();
    let core_validity = core_storage.validity()?;
    let shredded_validity = validity.clone();

    let masked_core_storage = match core_validity {
        Validity::NonNullable | Validity::AllValid => {
            // Core storage has no nulls, so wrap it in MaskedArray to apply the mask.
            MaskedArray::try_new(core_storage, validity)?.into_array()
        }
        Validity::AllInvalid => {
            // Already all-null, ANDing with any mask is still all-null.
            core_storage
        }
        Validity::Array(_) => {
            // Core storage already has nulls, but its physical validity layout depends on the
            // actual encoding. Use the mask operation instead of rewriting a presumed slot.
            core_storage.mask(validity.to_array(len))?
        }
    };
    let masked_shredded = if let Some(shredded) = array.shredded() {
        let canonical = shredded.clone().execute::<Canonical>(ctx)?;
        Some(mask_validity_canonical(canonical, shredded_validity, ctx)?.into_array())
    } else {
        None
    };

    VariantArray::try_new(masked_core_storage, masked_shredded)
}
