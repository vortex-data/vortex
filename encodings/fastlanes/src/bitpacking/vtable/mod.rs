// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use prost::Message;
use vortex_array::AnyCanonical;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_patches;
use vortex_array::require_validity;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::BitPackedData;
use crate::BitPackedDataParts;
use crate::bitpack_decompress::unpack_array;
use crate::bitpack_decompress::unpack_into_primitive_builder;
use crate::bitpacking::array::NUM_SLOTS;
use crate::bitpacking::array::PATCH_CHUNK_OFFSETS_SLOT;
use crate::bitpacking::array::PATCH_INDICES_SLOT;
use crate::bitpacking::array::PATCH_VALUES_SLOT;
use crate::bitpacking::array::SLOT_NAMES;
use crate::bitpacking::array::VALIDITY_SLOT;
use crate::bitpacking::vtable::kernels::PARENT_KERNELS;
use crate::bitpacking::vtable::rules::RULES;
mod kernels;
mod operations;
mod rules;
mod validity;

vtable!(BitPacked, BitPacked, BitPackedData);

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl VTable for BitPacked {
    type ArrayData = BitPackedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        data.validate_against_outer(dtype, len)
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &BitPackedData,
        state: &mut H,
        precision: Precision,
    ) {
        array.offset.hash(state);
        array.bit_width.hash(state);
        array.packed.array_hash(state, precision);
        option_array_hash(&array.slots[PATCH_INDICES_SLOT], state, precision);
        option_array_hash(&array.slots[PATCH_VALUES_SLOT], state, precision);
        option_array_hash(&array.slots[PATCH_CHUNK_OFFSETS_SLOT], state, precision);
        option_array_hash(&array.slots[VALIDITY_SLOT], state, precision);
        array.patch_offset.hash(state);
        array.patch_offset_within_chunk.hash(state);
    }

    fn array_eq(array: &BitPackedData, other: &BitPackedData, precision: Precision) -> bool {
        array.offset == other.offset
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
            && option_array_eq(
                &array.slots[PATCH_INDICES_SLOT],
                &other.slots[PATCH_INDICES_SLOT],
                precision,
            )
            && option_array_eq(
                &array.slots[PATCH_VALUES_SLOT],
                &other.slots[PATCH_VALUES_SLOT],
                precision,
            )
            && option_array_eq(
                &array.slots[PATCH_CHUNK_OFFSETS_SLOT],
                &other.slots[PATCH_CHUNK_OFFSETS_SLOT],
                precision,
            )
            && option_array_eq(
                &array.slots[VALIDITY_SLOT],
                &other.slots[VALIDITY_SLOT],
                precision,
            )
            && array.patch_offset == other.patch_offset
            && array.patch_offset_within_chunk == other.patch_offset_within_chunk
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.packed().clone(),
            _ => vortex_panic!("BitPackedArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("packed".to_string()),
            _ => None,
        }
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "BitPackedArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );

        // If patch slots are being cleared, clear the metadata too
        if slots[PATCH_INDICES_SLOT].is_none() || slots[PATCH_VALUES_SLOT].is_none() {
            array.patch_offset = None;
            array.patch_offset_within_chunk = None;
        }

        array.slots = slots;
        Ok(())
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BitPackedMetadata {
                bit_width: array.bit_width() as u32,
                offset: array.offset() as u32,
                patches: array
                    .patches(array.len())
                    .map(|p| p.to_metadata(array.len(), array.dtype()))
                    .transpose()?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<BitPackedData> {
        let metadata = BitPackedMetadata::decode(metadata)?;
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

        BitPackedData::try_new(
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
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            unpack_into_primitive_builder::<T>(
                &array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
                ctx,
            )
        })
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_patches!(
            array,
            array.patches(array.len()),
            PATCH_INDICES_SLOT,
            PATCH_VALUES_SLOT,
            PATCH_CHUNK_OFFSETS_SLOT
        );
        require_validity!(
            array,
            &array.validity(array.dtype().nullability()),
            VALIDITY_SLOT => AnyCanonical
        );

        Ok(ExecutionResult::done(
            unpack_array(array.as_view(), ctx)?.into_array(),
        ))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct BitPacked;

impl BitPacked {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.bitpacked");

    pub fn try_new(
        packed: BufferHandle,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        bit_width: u8,
        len: usize,
        offset: u16,
    ) -> VortexResult<BitPackedArray> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let data =
            BitPackedData::try_new(packed, ptype, validity, patches, bit_width, len, offset)?;
        Array::try_from_parts(ArrayParts::new(BitPacked, dtype, len, data))
    }

    pub fn into_parts(array: BitPackedArray) -> BitPackedDataParts {
        let len = array.len();
        let nullability = array.dtype().nullability();
        array.into_data().into_parts(len, nullability)
    }

    /// Encode an array into a bitpacked representation with the given bit width.
    pub fn encode(array: &ArrayRef, bit_width: u8) -> VortexResult<BitPackedArray> {
        BitPackedData::encode(array, bit_width)
    }
}

fn option_array_hash<H: std::hash::Hasher>(
    array: &Option<ArrayRef>,
    state: &mut H,
    precision: Precision,
) {
    match array {
        Some(array) => {
            true.hash(state);
            array.array_hash(state, precision);
        }
        None => false.hash(state),
    }
}

fn option_array_eq(lhs: &Option<ArrayRef>, rhs: &Option<ArrayRef>, precision: Precision) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.array_eq(rhs, precision),
        (None, None) => true,
        _ => false,
    }
}
