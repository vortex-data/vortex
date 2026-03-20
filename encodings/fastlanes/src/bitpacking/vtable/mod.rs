// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
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
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
use vortex_array::vtable::patches_child;
use vortex_array::vtable::patches_child_name;
use vortex_array::vtable::patches_nchildren;
use vortex_array::vtable::validity_nchildren;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::BitPackedArray;
use crate::bitpack_decompress::unpack_array;
use crate::bitpack_decompress::unpack_into_primitive_builder;
use crate::bitpacking::vtable::kernels::PARENT_KERNELS;
use crate::bitpacking::vtable::rules::RULES;
mod kernels;
mod operations;
mod rules;
mod validity;

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

impl VTable for BitPacked {
    type Array = BitPackedArray;

    type Metadata = ProstMetadata<BitPackedMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &BitPacked
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &BitPackedArray) -> usize {
        array.len
    }

    fn dtype(array: &BitPackedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BitPackedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &BitPackedArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.offset.hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
        array.bit_width.hash(state);
        array.packed.array_hash(state, precision);
        array.patches.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &BitPackedArray, other: &BitPackedArray, precision: Precision) -> bool {
        array.offset == other.offset
            && array.len == other.len
            && array.dtype == other.dtype
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
            && array.patches.array_eq(&other.patches, precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &BitPackedArray) -> usize {
        1
    }

    fn buffer(array: &BitPackedArray, idx: usize) -> BufferHandle {
        match idx {
            0 => array.packed().clone(),
            _ => vortex_panic!("BitPackedArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &BitPackedArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("packed".to_string()),
            _ => None,
        }
    }

    fn nchildren(array: &BitPackedArray) -> usize {
        array.patches().map_or(0, patches_nchildren) + validity_nchildren(&array.validity)
    }

    fn child(array: &BitPackedArray, idx: usize) -> ArrayRef {
        let pc = array.patches().map_or(0, patches_nchildren);
        if idx < pc {
            patches_child(
                array
                    .patches()
                    .vortex_expect("BitPackedArray child index out of bounds"),
                idx,
            )
        } else if idx < pc + validity_nchildren(&array.validity) {
            validity_to_child(&array.validity, array.len)
                .vortex_expect("BitPackedArray child index out of bounds")
        } else {
            vortex_panic!("BitPackedArray child index {idx} out of bounds")
        }
    }

    fn child_name(array: &BitPackedArray, idx: usize) -> String {
        let pc = array.patches().map_or(0, patches_nchildren);
        if idx < pc {
            patches_child_name(idx).to_string()
        } else {
            "validity".to_string()
        }
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(unpack_array(array, ctx)?.into_array()))
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Debug)]
pub struct BitPacked;

impl BitPacked {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.bitpacked");
}
