// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::AnyCanonical;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
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
use vortex_array::stats::ArrayStats;
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

    type Metadata = ProstMetadata<BitPackedMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &BitPackedData) -> &Self {
        &BitPacked
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &BitPackedData) -> usize {
        array.len
    }

    fn dtype(array: &BitPackedData) -> &DType {
        &array.dtype
    }

    fn stats(array: &BitPackedData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &BitPackedData,
        state: &mut H,
        precision: Precision,
    ) {
        array.offset.hash(state);
        array.bit_width.hash(state);
        array.packed.array_hash(state, precision);
        array.patches().array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &BitPackedData, other: &BitPackedData, precision: Precision) -> bool {
        array.offset == other.offset
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
            && array.patches().array_eq(&other.patches(), precision)
            && array.validity().array_eq(&other.validity(), precision)
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

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<BitPackedMetadata> as DeserializeMetadata>::deserialize(bytes)?;
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
    ) -> VortexResult<BitPackedData> {
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
        match_each_integer_ptype!(array.ptype(), |T| {
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
            array.patches(),
            PATCH_INDICES_SLOT,
            PATCH_VALUES_SLOT,
            PATCH_CHUNK_OFFSETS_SLOT
        );
        require_validity!(array, &array.validity(), VALIDITY_SLOT => AnyCanonical);

        Ok(ExecutionResult::done(
            unpack_array(&array, ctx)?.into_array(),
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

    /// Encode an array into a bitpacked representation with the given bit width.
    pub fn encode(array: &ArrayRef, bit_width: u8) -> VortexResult<BitPackedArray> {
        BitPackedData::encode(array, bit_width)
    }
}
