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
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::BitPackedDataParts;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedSlots;
use crate::bitpacking::array::BitPackedSlotsView;
use crate::bitpacking::array::PATCH_SLOTS;
use crate::bitpacking::array::validate_width_table;
use crate::bitpacking::array::width_table_child;
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

/// Metadata of the frozen `fastlanes.bitpacked` format: every chunk is packed at `bit_width`.
///
/// Arrays whose chunks differ in width serialize as `fastlanes.bitpacked_v2` through
/// `BitPackedPlugin`, which carries the widths in a child instead.
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
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.offset.hash(state);
        self.widths.hash(state);
        self.packed.array_hash(state, accuracy);
        self.patches_data.hash(state);
    }
}

impl ArrayEq for BitPackedData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.offset == other.offset
            && self.widths == other.widths
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
        let bp_slots = BitPackedSlotsView::from_slots(slots);

        let validity = child_to_validity(bp_slots.validity_child, dtype.nullability());
        let patches =
            PatchesData::patches_from_slots(data.patches_data.as_ref(), len, slots, PATCH_SLOTS);
        BitPackedData::validate(
            &data.packed,
            dtype.as_ptype(),
            &validity,
            patches.as_ref(),
            &data.widths,
            len,
            data.offset,
        )?;
        validate_width_table(&data.widths, bp_slots.width_table)
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
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // This output is labelled with the encoding's own ID, whose frozen contract is one width
        // for every chunk. Differing widths need the v2 format, which only `BitPackedPlugin`
        // emits.
        if !array.chunk_widths().is_uniform() {
            return Ok(None);
        }
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
        let packed = single_buffer(buffers)?;
        let offset = offset_from_metadata(metadata.offset)?;
        let bit_width = u8::try_from(metadata.bit_width).map_err(|_| {
            vortex_err!(
                "BitPackedMetadata bit_width {} does not fit in u8",
                metadata.bit_width
            )
        })?;
        let num_chunks = (len + offset as usize).div_ceil(FL_CHUNK_SIZE);
        let (patches, validity, _) =
            deserialize_children(children, metadata.patches, dtype, len, 0)?;

        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s.push(None);
            s
        };
        let widths = ChunkWidths::uniform(bit_width, num_chunks);
        let data = BitPackedData::try_new(packed, patches, widths, offset)?;
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

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

/// The single packed buffer of a serialized bit-packed array.
pub(crate) fn single_buffer(buffers: &[BufferHandle]) -> VortexResult<BufferHandle> {
    vortex_ensure!(
        buffers.len() == 1,
        "Expected 1 buffer, got {}",
        buffers.len()
    );
    Ok(buffers[0].clone())
}

/// The offset into the first chunk, which the metadata stores as a `u32`.
pub(crate) fn offset_from_metadata(offset: u32) -> VortexResult<u16> {
    u16::try_from(offset)
        .map_err(|_| vortex_err!("BitPackedMetadata offset {offset} does not fit in u16"))
}

/// Read the patches and validity children that both wire formats share.
///
/// Children run: the patches, then a validity bitmap if there is one, then `trailing` children
/// the caller reads itself. Returns the index of the first trailing child.
pub(crate) fn deserialize_children(
    children: &dyn ArrayChildren,
    patches: Option<PatchesMetadata>,
    dtype: &DType,
    len: usize,
    trailing: usize,
) -> VortexResult<(Option<Patches>, Validity, usize)> {
    let num_patch_children = match &patches {
        None => 0,
        Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
        Some(_) => 2,
    };
    let num_fixed = num_patch_children + trailing;
    let has_validity = match children.len().checked_sub(num_fixed) {
        Some(0) => false,
        Some(1) => true,
        _ => vortex_bail!(
            "Expected {num_fixed} or {} children, got {}",
            num_fixed + 1,
            children.len()
        ),
    };
    let validity = if has_validity {
        Validity::Array(children.get(num_patch_children, &Validity::DTYPE, len)?)
    } else {
        Validity::from(dtype.nullability())
    };
    let patches = patches
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
    Ok((
        patches,
        validity,
        num_patch_children + usize::from(has_validity),
    ))
}

#[derive(Clone, Debug)]
pub struct BitPacked;

impl BitPacked {
    /// Build a bit-packed array from its parts, with one width per chunk.
    pub fn try_new(
        packed: BufferHandle,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        widths: ChunkWidths,
        len: usize,
        offset: u16,
    ) -> VortexResult<BitPackedArray> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s.push(width_table_child(&widths));
            s
        };
        let data = BitPackedData::try_new(packed, patches, widths, offset)?;
        Array::try_from_parts(ArrayParts::new(BitPacked, dtype, len, data).with_slots(slots))
    }

    /// Replace the width table child of an array whose chunk widths differ with `table`, an
    /// array equal to the current table, so a compressor can re-encode it before the array is
    /// written. The table must hold one `u8` per chunk.
    pub fn with_width_table(
        array: BitPackedArray,
        table: ArrayRef,
    ) -> VortexResult<BitPackedArray> {
        let mut slots: ArraySlots = array.slots().iter().cloned().collect();
        slots[BitPackedSlots::WIDTH_TABLE] = Some(table);
        let dtype = array.dtype().clone();
        let len = array.len();
        let stats = array.statistics().to_owned();
        Ok(Array::try_from_parts(
            ArrayParts::new(BitPacked, dtype, len, array.into_data()).with_slots(slots),
        )?
        .with_stats_set(stats))
    }

    pub fn into_parts(array: BitPackedArray) -> BitPackedDataParts {
        let len = array.len();
        let patches = array.patches();
        let validity = array.validity().vortex_expect("BitPacked validity");
        let data = array.into_data();
        BitPackedDataParts {
            offset: data.offset,
            widths: data.widths,
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
