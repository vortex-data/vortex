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
use vortex_array::LEGACY_SESSION;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::PatchedArray;
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
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
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

    // NOTE(aduffy): Starting with format version 0.58.0, this field should never be set. It is
    //  only set by older writers and we use it to migrate to the new PatchedArray wrapper.
    #[prost(message, optional, tag = "3")]
    patches: Option<PatchesMetadata>,
}

impl BitPackedMetadata {
    pub(crate) fn new(bit_width: u32, offset: u32) -> Self {
        Self {
            bit_width,
            offset,
            patches: None,
        }
    }
}

impl VTable for BitPackedVTable {
    type Array = BitPackedArray;

    type Metadata = ProstMetadata<BitPackedMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn id(_array: &Self::Array) -> ArrayId {
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
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &BitPackedArray, other: &BitPackedArray, precision: Precision) -> bool {
        array.offset == other.offset
            && array.len == other.len
            && array.dtype == other.dtype
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
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
        validity_nchildren(&array.validity)
    }

    fn child(array: &BitPackedArray, idx: usize) -> ArrayRef {
        if idx < validity_nchildren(&array.validity) {
            validity_to_child(&array.validity, array.len)
                .vortex_expect("BitPackedArray child index out of bounds")
        } else {
            vortex_panic!("BitPackedArray child index {idx} out of bounds")
        }
    }

    fn child_name(array: &BitPackedArray, idx: usize) -> String {
        if idx < validity_nchildren(array.validity()) {
            return "validity".to_string();
        }

        vortex_panic!("invalid child index for BitPackedArray: {idx}");
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // Children: validity (if present).
        let expected_children = if matches!(array.validity, Validity::Array(_)) {
            1
        } else {
            0
        };

        vortex_ensure!(
            children.len() == expected_children,
            "expected {expected_children} children for BitPackedArray, received {}",
            children.len()
        );

        let validity = match children.into_iter().next() {
            Some(child) => Validity::Array(child),
            None => Validity::from(array.dtype.nullability()),
        };

        array.validity = validity;

        Ok(())
    }

    fn metadata(array: &BitPackedArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(BitPackedMetadata::new(
            array.bit_width as u32,
            array.offset() as u32,
        )))
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

        let bp_array = BitPackedArray::try_new(
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
        .into_array();

        if let Some(patches) = patches {
            // TODO(aduffy): this is only needed for backward compatibility.
            let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());
            Ok(
                PatchedArray::from_array_and_patches(bp_array.into_array(), &patches, &mut ctx)?
                    .into_array(),
            )
        } else {
            Ok(bp_array)
        }
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
pub struct BitPackedVTable;

impl BitPackedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.bitpacked");
}
