// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayEq;
use crate::ArrayHash;
mod kernels;
mod operations;
mod plugin;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::require_child;

/// A [`Patched`]-encoded Vortex array.
pub type PatchedArray = Array<Patched>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

#[derive(Clone, Debug)]
pub struct Patched;

impl ValidityChild<Patched> for Patched {
    fn validity_child(array: ArrayView<'_, Patched>) -> ArrayRef {
        array.inner().clone()
    }
}

/// Serialized metadata for `vortex.patched`.
///
/// The array has no buffers. Its children are, in order:
///
/// 1. `inner`: the outer dtype, with the outer length.
/// 2. `lane_offsets`: non-nullable `u32`, with `n_lanes` entries for each chunk of 1024 values
///    holding `offset + len` values, plus a final end offset.
/// 3. `patch_indices`: non-nullable `u16`, with `n_patches` entries.
/// 4. `patch_values`: the non-nullable outer dtype, with `n_patches` entries.
///
/// Readers require at least these children and ignore any that follow. A later writer may append
/// a child under this ID only if decoding without it still yields the same values, such as a
/// statistic or an index. A child or metadata field that changes the decoded values needs a new
/// encoding ID.
#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    /// The total number of patches, and the length of the indices and values child arrays.
    #[prost(uint32, tag = "1")]
    pub(crate) n_patches: u32,

    /// The number of lanes used for patch indexing. Must be a power of two between 1 and 128.
    #[prost(uint32, tag = "2")]
    pub(crate) n_lanes: u32,

    /// An offset into the first chunk's patches that should be considered in-view.
    ///
    /// Always less than 1024.
    #[prost(uint32, tag = "3")]
    pub(crate) offset: u32,
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
        self.n_lanes.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset && self.n_lanes == other.n_lanes
    }
}

impl VTable for Patched {
    type TypedArrayData = PatchedData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patched");
        *ID
    }

    fn validate(
        &self,
        data: &PatchedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, &PatchedSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.array().dtype();

        if !dtype.is_primitive() {
            // Default pathway: canonicalize and propagate.
            let canonical = array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array();
            return canonical.append_to_builder(builder, ctx);
        }

        let ptype = dtype.as_ptype();

        let len = array.len();

        array.inner().append_to_builder(builder, ctx)?;

        let offset = array.offset();
        let lane_offsets = array
            .lane_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;

        match_each_native_ptype!(ptype, |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Overwrite the last `len` elements of the builder. These would have been
            // populated by the inner.append_to_builder() call above.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;

            apply_patches_primitive::<V>(
                &mut output[trailer..],
                offset,
                len,
                array.n_lanes(),
                lane_offsets.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );
        });

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.inner(), PatchedSlots::INNER => Primitive);
        let array =
            require_child!(array, array.lane_offsets(), PatchedSlots::LANE_OFFSETS => Primitive);
        let array =
            require_child!(array, array.patch_indices(), PatchedSlots::PATCH_INDICES => Primitive);
        let array =
            require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => Primitive);

        let len = array.len();

        let n_lanes = array.n_lanes;
        let offset = array.offset;
        let slots = match array.try_into_parts() {
            Ok(parts) => PatchedSlots::from_slots(parts.slots),
            Err(array) => PatchedSlotsView::from_slots(array.slots()).to_owned(),
        };

        // TODO(joe): use iterative execution
        let PrimitiveDataParts {
            buffer,
            ptype,
            validity,
        } = slots.inner.downcast::<Primitive>().into_data_parts();

        let values = slots.patch_values.downcast::<Primitive>();
        let lane_offsets = slots.lane_offsets.downcast::<Primitive>();
        let patch_indices = slots.patch_indices.downcast::<Primitive>();

        let patched_values = match_each_native_ptype!(values.ptype(), |V| {
            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();

            apply_patches_primitive::<V>(
                &mut output,
                offset,
                len,
                n_lanes,
                lane_offsets.as_slice::<u32>(),
                patch_indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );

            let output = output.freeze();

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }
}

/// Apply patches on top of the existing value types.
fn apply_patches_primitive<V: NativePType>(
    output: &mut [V],
    offset: usize,
    len: usize,
    n_lanes: usize,
    lane_offsets: &[u32],
    indices: &[u16],
    values: &[V],
) {
    let n_chunks = (offset + len).div_ceil(1024);
    for chunk in 0..n_chunks {
        let start = lane_offsets[chunk * n_lanes] as usize;
        let stop = lane_offsets[chunk * n_lanes + n_lanes] as usize;

        for idx in start..stop {
            // the indices slice is measured as an offset into the 1024-value chunk.
            let index = chunk * 1024 + indices[idx] as usize;
            if index < offset || index >= offset + len {
                continue;
            }

            let value = values[idx];
            output[index - offset] = value;
        }
    }
}

#[cfg(test)]
mod tests;
