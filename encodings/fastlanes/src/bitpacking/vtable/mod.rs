// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::BitPackedArray;
use crate::bitpack_decompress::unpack_array;
use crate::bitpack_decompress::unpack_into_primitive_builder;
use crate::bitpacking::rules::RULES;
use crate::bitpacking::vtable::kernels::filter::PARENT_KERNELS;

mod array;
mod kernels;
mod operations;
mod validity;
mod visitor;

vtable!(BitPacked);

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl VTable for BitPackedVTable {
    type Array = BitPackedArray;

    type Metadata = ProstMetadata<BitPackedMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // Children: patches (if present): indices, values, chunk_offsets; then validity (if present)
        let patches_info = array
            .patches()
            .map(|p| (p.offset(), p.chunk_offsets().is_some()));

        let mut child_idx = 0;
        let patches = if let Some((patch_offset, has_chunk_offsets)) = patches_info {
            let patch_indices = children
                .get(child_idx)
                .ok_or_else(|| vortex_err!("Expected patch_indices child at index {}", child_idx))?
                .clone();
            child_idx += 1;

            let patch_values = children
                .get(child_idx)
                .ok_or_else(|| vortex_err!("Expected patch_values child at index {}", child_idx))?
                .clone();
            child_idx += 1;

            let patch_chunk_offsets = if has_chunk_offsets {
                let offsets = children
                    .get(child_idx)
                    .ok_or_else(|| {
                        vortex_err!("Expected patch_chunk_offsets child at index {}", child_idx)
                    })?
                    .clone();
                child_idx += 1;
                Some(offsets)
            } else {
                None
            };

            Some(Patches::new(
                array.len(),
                patch_offset,
                patch_indices,
                patch_values,
                patch_chunk_offsets,
            )?)
        } else {
            None
        };

        let validity = if child_idx < children.len() {
            Validity::Array(children[child_idx].clone())
        } else {
            Validity::from(array.dtype().nullability())
        };

        let expected_children = child_idx
            + if matches!(validity, Validity::Array(_)) {
                1
            } else {
                0
            };
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {} children, got {}",
            expected_children,
            children.len()
        );

        array.patches = patches;
        array.validity = validity;

        Ok(())
    }

    fn metadata(array: &BitPackedArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(BitPackedMetadata {
            bit_width: array.bit_width() as u32,
            offset: array.offset() as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()?,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<BitPackedMetadata> as DeserializeMetadata>::deserialize(buffer)?;
        Ok(ProstMetadata(inner))
    }

    /// Deserialize a BitPackedArray from its components.
    ///
    /// Note that the layout depends on whether patches and chunk_offsets are present:
    /// - No patches: `[validity?]`
    /// - With patches: `[patch_indices, patch_values, chunk_offsets?, validity?]`
    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<BitPackedArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let packed = buffers[0].clone();

        let load_validity = |child_idx: usize| {
            if children.len() == child_idx {
                Ok(Validity::from(dtype.nullability()))
            } else if children.len() == child_idx + 1 {
                let validity = children.get(child_idx, &Validity::DTYPE, len)?;
                Ok(Validity::Array(validity))
            } else {
                vortex_bail!(
                    "Expected {} or {} children, got {}",
                    child_idx,
                    child_idx + 1,
                    children.len()
                );
            }
        };

        let validity_idx = match &metadata.patches {
            None => 0,
            Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
            Some(_) => 2,
        };

        let validity = load_validity(validity_idx)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(0, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(1, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                    .transpose()?;

                Patches::new(len, p.offset()?, indices, values, chunk_offsets)
            })
            .transpose()?;

        BitPackedArray::try_new(
            packed,
            PType::try_from(dtype)?,
            validity,
            patches,
            u8::try_from(metadata.bit_width).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata bit_width {} does not fit in u8",
                    metadata.bit_width
                )
            })?,
            len,
            u16::try_from(metadata.offset).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata offset {} does not fit in u16",
                    metadata.offset
                )
            })?,
        )
    }

    fn append_to_builder(
        array: &BitPackedArray,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.ptype(), |T| {
            unpack_into_primitive_builder::<T>(
                array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
                ctx,
            )
        })
    }

    fn canonicalize(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(unpack_array(array, ctx)?))
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    // TODO(joe): fix me https://github.com/vortex-data/vortex/pull/5958#discussion_r2696436008
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let offset_start = range.start + array.offset() as usize;
        let offset_stop = range.end + array.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = offset_stop.div_ceil(1024) * 1024;

        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;

        // slice the buffer using the encoded start/stop values
        // SAFETY: slicing packed values without decoding preserves invariants
        Ok(Some(unsafe {
            BitPackedArray::new_unchecked(
                array.packed().slice(encoded_start..encoded_stop),
                array.dtype.clone(),
                array.validity().slice(range.clone())?,
                array
                    .patches()
                    .map(|p| p.slice(range.clone()))
                    .transpose()?
                    .flatten(),
                array.bit_width(),
                range.len(),
                offset as u16,
            )
            .into_array()
        }))
    }
}

#[derive(Debug)]
pub struct BitPackedVTable;

impl BitPackedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.bitpacked");
}
