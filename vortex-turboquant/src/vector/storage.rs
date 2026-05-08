// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant physical storage helpers.
//!
//! TurboQuant storage is row-aligned and full length:
//!
//! ```text
//! Struct {
//!     norms: Primitive<element_ptype, vector_validity>,
//!     codes: FixedSizeList<Primitive<u8>, padded_dim, vector_validity>,
//! }
//! ```
//!
//! Row nullability is carried on the outer struct and on the `norms` and `codes` field arrays.
//! This is deliberate duplication: null vectors remain null throughout encode/decode instead of being
//! converted into zero vectors. The code bytes for invalid rows are physical placeholders only; the
//! field-level validity records that those rows were not quantized.
//!
//! Parsing treats the outer struct validity as authoritative. Child validity may be wider than the
//! struct validity, for example after a generic mask only updates the struct validity, but each
//! child must be valid wherever the struct row is valid.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::quantize::QuantizationResult;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_metadata;

/// Name of the stored row-norm child.
pub(crate) const NORMS_FIELD: &str = "norms";

/// Name of the stored quantized-code child.
pub(crate) const CODES_FIELD: &str = "codes";

/// Parsed TurboQuant storage arrays.
///
/// We use this as a helper struct for working with a TurboQuant extension array.
pub(crate) struct TurboQuantParsedStorage {
    pub(crate) metadata: TurboQuantMetadata,
    pub(crate) vector_validity: Validity,
    pub(crate) norms: PrimitiveArray,
    pub(crate) codes: PrimitiveArray,
    pub(crate) len: usize,
}

/// Build the `codes: FixedSizeList<Primitive<u8>, padded_dim>` storage child.
///
/// Each row of `padded_dim` u8 codes indexes into the deterministic centroid codebook derived
/// from `(padded_dim, bit_width)`. The centroid values are intentionally not stored in the array.
pub(crate) fn build_codes_child(
    num_vectors: usize,
    quantization: QuantizationResult,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    let codes = PrimitiveArray::new::<u8>(quantization.all_indices, Validity::NonNullable);
    let padded_dim_u32 = u32::try_from(quantization.padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;

    Ok(FixedSizeListArray::try_new(
        codes.into_array(),
        padded_dim_u32,
        vector_validity,
        num_vectors,
    )?
    .into_array())
}

/// Build the TurboQuant `Struct { norms, codes }` storage array.
pub(crate) fn build_storage(
    norms: ArrayRef,
    codes: ArrayRef,
    len: usize,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(StructArray::try_new(
        FieldNames::from([NORMS_FIELD, CODES_FIELD]),
        vec![norms, codes],
        len,
        vector_validity,
    )?
    .into_array())
}

/// Parse a TurboQuant extension array into executed storage children.
pub(crate) fn parse_storage(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TurboQuantParsedStorage> {
    let metadata = tq_metadata(input.dtype())?;
    let ext: ExtensionArray = input.execute(ctx)?;
    let storage: StructArray = ext.storage_array().clone().execute(ctx)?;

    let norms: PrimitiveArray = storage
        .unmasked_field_by_name(NORMS_FIELD)?
        .clone()
        .execute(ctx)?;

    let codes_fsl: FixedSizeListArray = storage
        .unmasked_field_by_name(CODES_FIELD)?
        .clone()
        .execute(ctx)?;
    let codes: PrimitiveArray = codes_fsl.elements().clone().execute(ctx)?;

    let len = storage.len();
    let struct_validity = storage.struct_validity();
    let norms_validity = norms.validity()?;
    let codes_validity = codes_fsl.validity()?;

    let struct_mask = struct_validity.execute_mask(len, ctx)?;
    let norms_mask = norms_validity.execute_mask(len, ctx)?;
    let codes_mask = codes_validity.execute_mask(len, ctx)?;
    validate_child_validity_covers_struct(&struct_mask, &norms_mask, &codes_mask)?;

    Ok(TurboQuantParsedStorage {
        metadata,
        vector_validity: struct_validity,
        norms,
        codes,
        len,
    })
}

/// Validate that both child masks cover the struct mask: every row that the struct considers
/// valid must also be valid in the `norms` and `codes` children.
///
/// `struct_mask & !child_mask` selects rows where the struct is valid but the child is not. If
/// no such row exists, the child covers the struct. [`Mask::bitand_not`] is variant-specialized,
/// so this short-circuits in `O(1)` when either mask is `AllTrue` or `AllFalse`.
fn validate_child_validity_covers_struct(
    struct_mask: &Mask,
    norms_mask: &Mask,
    codes_mask: &Mask,
) -> VortexResult<()> {
    vortex_ensure!(
        struct_mask.clone().bitand_not(norms_mask).all_false(),
        "TurboQuant {NORMS_FIELD} row validity must cover storage validity"
    );
    vortex_ensure!(
        struct_mask.clone().bitand_not(codes_mask).all_false(),
        "TurboQuant {CODES_FIELD} row validity must cover storage validity"
    );
    Ok(())
}
