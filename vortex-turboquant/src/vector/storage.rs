// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant physical storage helpers.
//!
//! TurboQuant storage is row-aligned and full length:
//!
//! ```text
//! Struct {
//!     norms: Primitive<element_ptype, vector_validity>,
//!     inv_direction_norms: Primitive<f32, vector_validity>,
//!     codes: FixedSizeList<Primitive<u8>, padded_dim, vector_validity>,
//! }
//! ```
//!
//! `inv_direction_norms` is pinned to `f32` regardless of `element_ptype` because the SORF
//! transform and the centroid codebook are both `f32`; storing it wider would add precision the
//! underlying computation does not have.
//!
//! Row nullability is carried on the outer struct AND on every row-aligned field array. This is
//! deliberate duplication: null vectors remain null throughout encode/decode instead of being
//! converted into zero vectors. The code bytes and inverse direction norms for invalid rows are
//! physical placeholders only; the field-level validity records that those rows were not
//! quantized.
//!
//! Parsing treats the outer struct validity as authoritative. Child validity may be wider than
//! the struct validity (for example after a generic mask only updates the struct validity), but
//! each child must be valid wherever the struct row is valid.

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
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_metadata;

/// Name of the stored row-norm child.
pub(crate) const NORMS_FIELD: &str = "norms";

/// Name of the stored inverse quantized-direction norm child.
pub(crate) const INV_DIRECTION_NORMS_FIELD: &str = "inv_direction_norms";

/// Name of the stored quantized-code child.
pub(crate) const CODES_FIELD: &str = "codes";

/// Executed storage children of a TurboQuant extension array plus the authoritative outer
/// struct validity. Every child is row-aligned to `len` and every child's validity covers
/// `vector_validity`.
pub(crate) struct TurboQuantParsedStorage {
    /// Metadata recovered from the input extension dtype.
    pub(crate) metadata: TurboQuantMetadata,
    /// Authoritative row validity for the quantized vectors, taken from the outer struct.
    pub(crate) vector_validity: Validity,
    /// Per-row stored L2 norm of the original input vector, in `metadata.element_ptype`.
    pub(crate) norms: PrimitiveArray,
    /// Per-row reciprocal L2 norm of the decoded direction (always `f32`). Multiplied through
    /// in `TQDecode` so that `L2Norm(TQDecode(_))` preserves the stored row norm. A stored
    /// `0.0` is a sentinel telling decode to emit an all-zero row; it pairs with a stored
    /// norm of `0.0` for valid zero-norm input rows and for the rare denormal-cancellation
    /// case (encode rejects non-finite input norms up front, so those cannot reach this
    /// field).
    pub(crate) inv_direction_norms: PrimitiveArray,
    /// Flat `u8` per-row centroid indices, `num_vectors * padded_dim` entries long.
    pub(crate) codes: PrimitiveArray,
    /// Row count.
    pub(crate) len: usize,
}

/// Subset of [`TurboQuantParsedStorage`] containing only the `norms` child plus the outer
/// struct validity. Used by the `L2Norm(TQDecode(_))` execute-parent kernel, which has no need
/// for the `codes` or `inv_direction_norms` children.
pub(crate) struct TurboQuantParsedNorms {
    /// Authoritative row validity for the quantized vectors.
    pub(crate) vector_validity: Validity,
    /// Per-row stored L2 norm of the original input vector, in `metadata.element_ptype`.
    pub(crate) norms: PrimitiveArray,
}

/// Build the `codes: FixedSizeList<Primitive<u8>, padded_dim>` storage child.
///
/// Each row of `padded_dim` u8 codes indexes into the deterministic centroid codebook derived
/// from `(padded_dim, bit_width)`. The centroid values are intentionally not stored in the array.
pub(crate) fn build_codes_child(
    num_vectors: usize,
    all_indices: Buffer<u8>,
    padded_dim: usize,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    let codes = PrimitiveArray::new::<u8>(all_indices, Validity::NonNullable);
    let padded_dim_u32 = u32::try_from(padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;

    Ok(FixedSizeListArray::try_new(
        codes.into_array(),
        padded_dim_u32,
        vector_validity,
        num_vectors,
    )?
    .into_array())
}

/// Build the TurboQuant `Struct { norms, inv_direction_norms, codes }` storage array.
pub(crate) fn build_storage(
    norms: ArrayRef,
    inv_direction_norms: ArrayRef,
    codes: ArrayRef,
    len: usize,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(StructArray::try_new(
        FieldNames::from([NORMS_FIELD, INV_DIRECTION_NORMS_FIELD, CODES_FIELD]),
        vec![norms, inv_direction_norms, codes],
        len,
        vector_validity,
    )?
    .into_array())
}

/// Parse a TurboQuant extension array into executed storage children.
///
/// Executes all three storage children, validates that every child's row validity covers the
/// outer struct validity, and returns the parsed result. Used by `TQDecode`, which needs every
/// child. Kernels that only need a subset should use a narrower helper (for example
/// [`parse_storage_norms_only`]) to avoid executing the children they will not consume.
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

    let inv_direction_norms: PrimitiveArray = storage
        .unmasked_field_by_name(INV_DIRECTION_NORMS_FIELD)?
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
    let inv_direction_norms_validity = inv_direction_norms.validity()?;
    let codes_validity = codes_fsl.validity()?;

    let struct_mask = struct_validity.execute_mask(len, ctx)?;
    let norms_mask = norms_validity.execute_mask(len, ctx)?;
    let inv_direction_norms_mask = inv_direction_norms_validity.execute_mask(len, ctx)?;
    let codes_mask = codes_validity.execute_mask(len, ctx)?;
    validate_child_validity_covers_struct(
        &struct_mask,
        &norms_mask,
        &inv_direction_norms_mask,
        &codes_mask,
    )?;

    Ok(TurboQuantParsedStorage {
        metadata,
        vector_validity: struct_validity,
        norms,
        inv_direction_norms,
        codes,
        len,
    })
}

/// Narrow form of [`parse_storage`] that returns only the `norms` child plus the outer struct
/// validity. Used by the `L2Norm(TQDecode(_))` kernel so the fast path does not execute the
/// `codes` and `inv_direction_norms` children it has no use for. The `norms` child's validity
/// is still validated against the struct's; the other children are not.
pub(crate) fn parse_storage_norms_only(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TurboQuantParsedNorms> {
    let ext: ExtensionArray = input.execute(ctx)?;
    let storage: StructArray = ext.storage_array().clone().execute(ctx)?;

    let norms: PrimitiveArray = storage
        .unmasked_field_by_name(NORMS_FIELD)?
        .clone()
        .execute(ctx)?;

    let len = storage.len();
    let struct_validity = storage.struct_validity();
    let norms_validity = norms.validity()?;

    let struct_mask = struct_validity.execute_mask(len, ctx)?;
    let norms_mask = norms_validity.execute_mask(len, ctx)?;
    vortex_ensure!(
        struct_mask.bitand_not(&norms_mask).all_false(),
        "TurboQuant {NORMS_FIELD} row validity must cover storage validity"
    );

    Ok(TurboQuantParsedNorms {
        vector_validity: struct_validity,
        norms,
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
    inv_direction_norms_mask: &Mask,
    codes_mask: &Mask,
) -> VortexResult<()> {
    vortex_ensure!(
        struct_mask.clone().bitand_not(norms_mask).all_false(),
        "TurboQuant {NORMS_FIELD} row validity must cover storage validity"
    );
    vortex_ensure!(
        struct_mask
            .clone()
            .bitand_not(inv_direction_norms_mask)
            .all_false(),
        "TurboQuant {INV_DIRECTION_NORMS_FIELD} row validity must cover storage validity"
    );
    vortex_ensure!(
        struct_mask.clone().bitand_not(codes_mask).all_false(),
        "TurboQuant {CODES_FIELD} row validity must cover storage validity"
    );
    Ok(())
}
