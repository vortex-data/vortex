// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedData;
use crate::BitPackedDataParts;
use crate::bitpacking::array::BitPackedSlots;
use crate::bitpacking::array::BitPackedSlotsView;
use crate::bitpacking::array::PATCH_SLOTS;
use crate::bitpacking::bitpack_decompress::unpack_array;
use crate::bitpacking::bitpack_decompress::unpack_into_primitive_builder;
use crate::bitpacking::vtable::rules::RULES;
mod kernels;
mod operations;
mod rules;
mod validity;

/// A [`BitPacked`]-encoded Vortex array.
pub type BitPackedArray = Array<BitPacked>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

impl ArrayHash for BitPackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.offset.hash(state);
        self.packed.array_hash(state, accuracy);
        self.patches_data.hash(state);
    }
}

impl ArrayEq for BitPackedData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.offset == other.offset
            && self.packed.array_eq(&other.packed, accuracy)
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
        vortex_ensure!(
            slots.len() == BitPackedSlots::COUNT,
            "Expected {} slots, got {}",
            BitPackedSlots::COUNT,
            slots.len()
        );
        vortex_ensure!(
            slots[BitPackedSlots::WIDTH_TABLE].is_some()
                && slots[BitPackedSlots::CHUNK_OFFSETS].is_some(),
            "Missing width table or chunk offsets"
        );
        let bp_slots = BitPackedSlotsView::from_slots(slots);

        let validity = child_to_validity(bp_slots.validity_child, dtype.nullability());
        let patches =
            PatchesData::patches_from_slots(data.patches_data.as_ref(), len, slots, PATCH_SLOTS);
        data.validate(
            dtype.as_ptype(),
            &validity,
            patches.as_ref(),
            bp_slots.width_table,
            bp_slots.chunk_offsets,
            len,
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.packed = buffers[0].clone();
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("BitPacked serialization requires BitPackedPlugin")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("BitPacked deserialization requires BitPackedPlugin")
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
    /// Build a bit-packed array with one width per chunk and a trailing byte-offset boundary.
    /// Offsets may have a nonzero origin, which is subtracted when indexing the packed buffer.
    #[expect(
        clippy::too_many_arguments,
        reason = "Each physical component of the encoding is explicit"
    )]
    pub fn try_new(
        packed: BufferHandle,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        widths: ArrayRef,
        chunk_offsets: ArrayRef,
        len: usize,
        offset: u16,
    ) -> VortexResult<BitPackedArray> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s.push(Some(widths));
            s.push(Some(chunk_offsets));
            s
        };
        let data = BitPackedData::try_new(packed, patches, offset)?;
        Array::try_from_parts(ArrayParts::new(BitPacked, dtype, len, data).with_slots(slots))
    }

    /// Replace the width table, preserving the offsets. Values must agree with the offsets.
    /// Value-dependent validation of compressed children is deferred until execution.
    pub fn with_width_table(
        array: BitPackedArray,
        table: ArrayRef,
    ) -> VortexResult<BitPackedArray> {
        let offsets = array.chunk_offsets().clone();
        Self::with_chunk_layout(array, table, offsets)
    }

    /// Replace both chunk-layout children. Widths must be non-nullable `u8`, and offsets
    /// non-nullable `u64`, with adjacent differences equal to `128 * width`.
    pub fn with_chunk_layout(
        array: BitPackedArray,
        widths: ArrayRef,
        offsets: ArrayRef,
    ) -> VortexResult<BitPackedArray> {
        let mut slots: ArraySlots = array.slots().iter().cloned().collect();
        slots[BitPackedSlots::WIDTH_TABLE] = Some(widths);
        slots[BitPackedSlots::CHUNK_OFFSETS] = Some(offsets);
        let dtype = array.dtype().clone();
        let len = array.len();
        Array::try_from_parts(
            ArrayParts::new(BitPacked, dtype, len, array.into_data()).with_slots(slots),
        )
    }

    pub fn into_parts(array: BitPackedArray) -> BitPackedDataParts {
        let len = array.len();
        let patches = array.patches();
        let validity = array.validity().vortex_expect("BitPacked validity");
        let widths = array.width_table().clone();
        let chunk_offsets = array.chunk_offsets().clone();
        let data = array.into_data();
        BitPackedDataParts {
            offset: data.offset,
            widths,
            chunk_offsets,
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
