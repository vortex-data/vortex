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
use vortex_array::patches::PatchesMetadata;
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
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &BitPackedData, other: &BitPackedData, precision: Precision) -> bool {
        array.offset == other.offset
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
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

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(BitPackedMetadata {
            bit_width: array.bit_width() as u32,
            offset: array.offset() as u32,
            patches: None,
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

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.ptype(), |T| {
            unpack_into_primitive_builder::<T>(
                array.data(),
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
            )
        })
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
    ) -> VortexResult<ArrayRef> {
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

        Ok(BitPackedData::try_new(
            packed,
            PType::try_from(dtype)?,
            validity,
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
        )?
        .into_array())
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

        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_validity!(array, &array.validity(), VALIDITY_SLOT => AnyCanonical);

        Ok(ExecutionResult::done(
            unpack_array(array.data(), ctx)?.into_array(),
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
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.bitpacked");
}
