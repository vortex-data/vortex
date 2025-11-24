// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::execution::ExecutionCtx;
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use vortex_array::{
    DeserializeMetadata, EncodingId, EncodingRef, ProstMetadata, SerializeMetadata, vtable,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_vector::{Vector, VectorMutOps};

use crate::BitPackedArray;
use crate::bitpack_decompress::unpack_to_primitive_vector;

mod array;
mod canonical;
mod encode;
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
    type Encoding = BitPackedEncoding;
    type Metadata = ProstMetadata<BitPackedMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.bitpacked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BitPackedEncoding.as_ref())
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
        _encoding: &BitPackedEncoding,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
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
            Some(patches_meta) if patches_meta.chunk_offsets_dtype().is_some() => 3,
            Some(_) => 2,
        };

        let validity = load_validity(validity_idx)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(0, &p.indices_dtype(), p.len())?;
                let values = children.get(1, dtype, p.len())?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()
                    .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                    .transpose()?;

                Ok::<_, VortexError>(Patches::new(
                    len,
                    p.offset(),
                    indices,
                    values,
                    chunk_offsets,
                ))
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

    fn execute(array: &BitPackedArray, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(unpack_to_primitive_vector(array).freeze().into())
    }
}

#[derive(Clone, Debug)]
pub struct BitPackedEncoding;
