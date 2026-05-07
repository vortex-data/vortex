// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
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
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_patches;
use vortex_array::require_validity;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::BitPackedDataParts;
use crate::bitpack_decompress::unpack_array;
use crate::bitpack_decompress::unpack_into_primitive_builder;
use crate::bitpacking::array::BitPackedSlots;
use crate::bitpacking::array::BitPackedSlotsView;
use crate::bitpacking::array::PATCH_SLOTS;
use crate::bitpacking::vtable::kernels::PARENT_KERNELS;
use crate::bitpacking::vtable::rules::RULES;
mod kernels;
mod operations;
mod rules;
mod validity;

/// A [`BitPacked`]-encoded Vortex array.
pub type BitPackedArray = Array<BitPacked>;

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ArrayHash for BitPackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.offset.hash(state);
        self.bit_width.hash(state);
        self.packed.array_hash(state, precision);
        self.patches_data.hash(state);
    }
}

impl ArrayEq for BitPackedData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.offset == other.offset
            && self.bit_width == other.bit_width
            && self.packed.array_eq(&other.packed, precision)
            && self.patches_data == other.patches_data
    }
}

impl VTable for BitPacked {
    type TypedArrayData = BitPackedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.bitpacked");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let bp_slots = BitPackedSlotsView::from_slots(slots);

        let validity = child_to_validity(bp_slots.validity_child, dtype.nullability());
        let patches =
            PatchesData::patches_from_slots(data.patches_data.as_ref(), len, slots, PATCH_SLOTS);
        BitPackedData::validate(
            &data.packed,
            dtype.as_ptype(),
            &validity,
            patches.as_ref(),
            data.bit_width,
            len,
            data.offset,
        )
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BitPackedMetadata {
                bit_width: array.bit_width() as u32,
                offset: array.offset() as u32,
                patches: array
                    .patches()
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
    ) -> VortexResult<ArrayParts<Self>> {
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

        let slots = {
            let mut s = ArraySlots::with_capacity(4);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s
        };
        let data = BitPackedData::try_new(
            packed,
            patches,
            u8::try_from(metadata.bit_width).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata bit_width {} does not fit in u8",
                    metadata.bit_width
                )
            })?,
            u16::try_from(metadata.offset).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata offset {} does not fit in u16",
                    metadata.offset
                )
            })?,
        )?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BitPackedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_patches!(
            array,
            BitPackedSlots::PATCH_INDICES,
            BitPackedSlots::PATCH_VALUES,
            BitPackedSlots::PATCH_CHUNK_OFFSETS
        );
        require_validity!(array, BitPackedSlots::VALIDITY_CHILD);

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

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct BitPacked;

impl BitPacked {
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
        let slots = {
            let mut s = ArraySlots::with_capacity(4);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s
        };
        let data = BitPackedData::try_new(packed, patches, bit_width, offset)?;
        Array::try_from_parts(ArrayParts::new(BitPacked, dtype, len, data).with_slots(slots))
    }

    pub fn into_parts(array: BitPackedArray) -> BitPackedDataParts {
        let len = array.len();
        let patches = array.patches();
        let validity = array.validity().vortex_expect("BitPacked validity");
        let data = array.into_data();
        BitPackedDataParts {
            offset: data.offset,
            bit_width: data.bit_width,
            len,
            packed: data.packed,
            patches,
            validity,
        }
    }

    /// Encode an array into a bitpacked representation with the given bit width.
    pub fn encode(
        array: &ArrayRef,
        bit_width: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BitPackedArray> {
        BitPackedData::encode(array, bit_width, ctx)
    }
}
