// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant physical storage helpers.
//!
//! Block-decomposed TurboQuant storage is a row-aligned outer struct of inner
//! `Struct { norms, codes }` blocks, one per power-of-two block size in `metadata.block_sizes`:
//!
//! ```text
//! Struct {
//!     block_0: Struct {
//!         norms: Primitive<element_ptype, vector_validity>,
//!         codes: FixedSizeList<Primitive<u8>, block_sizes[0], vector_validity>,
//!     },
//!     ...
//!     block_{N-1}: Struct { norms: ..., codes: FixedSizeList<u8, block_sizes[N-1], ...> },
//! }
//! ```
//!
//! Outer struct validity is authoritative. Each inner block's struct validity must cover the outer.
//! Additionally, each inner block's `norms` and `codes` validity must cover the inner struct.

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
use vortex_error::vortex_ensure_eq;

use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_metadata;

/// Name of the stored row-norm child inside an inner block struct.
pub(crate) const NORMS_FIELD: &str = "norms";

/// Name of the stored quantized-code child inside an inner block struct.
pub(crate) const CODES_FIELD: &str = "codes";

/// Deterministic field name for the inner struct of block index `index`.
pub(crate) fn block_field_name(index: usize) -> String {
    format!("block_{index}")
}

/// The stored `(norms, codes)` of a single block.
///
/// Encode produces these from the quantized rows. Decode recovers them by executing and unwrapping
/// the physical storage.
pub(crate) struct Block {
    /// Per-row stored block L2 norm, in `metadata.element_ptype`.
    pub(crate) norms: PrimitiveArray,

    /// Flat per-row centroid indices, `num_vectors * block_sizes[i]` bytes long. Indexed as
    /// `codes[row * block_sizes[i] + j]`.
    ///
    /// The codes are flat here and only wrapped into a `FixedSizeList` by [`build_storage`] (and
    /// unwrapped back to flat by [`parse_storage`]).
    pub(crate) codes: PrimitiveArray,
}

/// Executed storage of a TurboQuant extension array, decomposed into per-block children plus the
/// authoritative outer struct validity. Every child is row-aligned to `len` and every inner-block
/// child's validity covers `vector_validity`.
pub(crate) struct TurboQuantParsedStorage {
    /// Metadata recovered from the input extension dtype.
    pub(crate) metadata: TurboQuantMetadata,

    /// Authoritative row validity, taken from the outer struct.
    pub(crate) vector_validity: Validity,

    /// One [`Block`] per entry in `metadata.block_sizes`, in order.
    pub(crate) blocks: Vec<Block>,

    /// Row count.
    pub(crate) len: usize,
}

/// Build the outer TurboQuant storage array from per-block encoder output.
///
/// `blocks` must have one entry per block in `block_sizes`, in block order. Each block's flat codes
/// are wrapped into a `FixedSizeList` of the block's width and paired with its norms in a
/// `Struct { norms, codes }` field of the outer struct.
pub(crate) fn build_storage(
    blocks: Vec<Block>,
    block_sizes: &[u32],
    num_vectors: usize,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    let mut names = Vec::with_capacity(blocks.len());
    let mut fields = Vec::with_capacity(blocks.len());

    for (index, (block, &block_size)) in blocks.into_iter().zip(block_sizes.iter()).enumerate() {
        names.push(block_field_name(index));

        let codes_fsl = FixedSizeListArray::try_new(
            block.codes.into_array(),
            block_size,
            vector_validity.clone(),
            num_vectors,
        )?
        .into_array();
        let inner = StructArray::try_new(
            FieldNames::from([NORMS_FIELD, CODES_FIELD]),
            vec![block.norms.into_array(), codes_fsl],
            num_vectors,
            vector_validity.clone(),
        )?
        .into_array();
        fields.push(inner);
    }

    Ok(StructArray::try_new(
        FieldNames::from_iter(names),
        fields,
        num_vectors,
        vector_validity,
    )?
    .into_array())
}

/// Parse a TurboQuant extension array into per-block executed storage children.
pub(crate) fn parse_storage(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TurboQuantParsedStorage> {
    let metadata = tq_metadata(input.dtype())?;
    let ext: ExtensionArray = input.execute(ctx)?;
    let outer: StructArray = ext.storage_array().clone().execute(ctx)?;

    let len = outer.len();
    let outer_validity = outer.struct_validity();
    let outer_mask = outer_validity.execute_mask(len, ctx)?;

    let mut blocks = Vec::with_capacity(metadata.block_sizes.len());
    for (index, &block) in metadata.block_sizes.iter().enumerate() {
        let name = block_field_name(index);
        let inner: StructArray = outer.unmasked_field_by_name(&name)?.clone().execute(ctx)?;

        // Ensure the outer struct mask covers the block mask.
        let inner_validity = inner.struct_validity();
        let inner_mask = inner_validity.execute_mask(len, ctx)?;
        vortex_ensure!(outer_mask.clone().bitand_not(&inner_mask).all_false());

        let norms: PrimitiveArray = inner
            .unmasked_field_by_name(NORMS_FIELD)?
            .clone()
            .execute(ctx)?;
        let codes_fsl: FixedSizeListArray = inner
            .unmasked_field_by_name(CODES_FIELD)?
            .clone()
            .execute(ctx)?;
        vortex_ensure_eq!(
            codes_fsl.list_size(),
            block,
            "TurboQuant inner block {name} {CODES_FIELD} list size must be {block}, got {}",
            codes_fsl.list_size()
        );
        let codes: PrimitiveArray = codes_fsl.elements().clone().execute(ctx)?;

        // Ensure that block mask covers the norms and codes masks.
        let norms_mask = norms.validity()?.execute_mask(len, ctx)?;
        let codes_mask = codes_fsl.validity()?.execute_mask(len, ctx)?;
        vortex_ensure!(inner_mask.clone().bitand_not(&norms_mask).all_false());
        vortex_ensure!(inner_mask.clone().bitand_not(&codes_mask).all_false());

        blocks.push(Block { norms, codes });
    }

    Ok(TurboQuantParsedStorage {
        metadata,
        vector_validity: outer_validity,
        blocks,
        len,
    })
}
