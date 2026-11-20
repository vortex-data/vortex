// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Patched`].

use prost::Message;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::PatchedMetadata;
use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::validity::Validity;

impl ArrayPlugin for Patched {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "Patched plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<Patched>();
        let mut ctx = session.create_execution_ctx();

        // A sliced array keeps every patch, but references only those of its own chunks. Write
        // just those, with the lane offsets rebased onto them.
        let n_offsets = array.lane_offsets().len();
        let first = lane_offset(array.lane_offsets(), 0, &mut ctx)?;
        let last = lane_offset(array.lane_offsets(), n_offsets - 1, &mut ctx)?;
        let (lane_offsets, indices, values) = if first == 0 && last == array.patch_indices().len() {
            (
                array.lane_offsets().clone(),
                array.patch_indices().clone(),
                array.patch_values().clone(),
            )
        } else {
            let lane_offsets = array
                .lane_offsets()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            let first = u32::try_from(first)?;
            let rebased: Buffer<u32> = lane_offsets
                .as_slice::<u32>()
                .iter()
                .map(|offset| offset - first)
                .collect();
            let range = first as usize..last;
            (
                PrimitiveArray::new(rebased, Validity::NonNullable).into_array(),
                array
                    .patch_indices()
                    .slice(range.clone())?
                    .optimize_ctx(&ctx)?,
                array.patch_values().slice(range)?.optimize_ctx(&ctx)?,
            )
        };

        let metadata = PatchedMetadata {
            n_patches: u32::try_from(indices.len())?,
            n_lanes: u32::try_from(array.n_lanes())?,
            offset: u32::try_from(array.offset())?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::new(
            ArrayPlugin::id(self),
            metadata,
            vec![],
            vec![array.inner().clone(), lane_offsets, indices, values],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Patched plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        let metadata = PatchedMetadata::decode(metadata)?;
        let n_patches = metadata.n_patches as usize;
        let n_lanes = metadata.n_lanes as usize;
        let offset = metadata.offset as usize;

        // n_chunks should correspond to the chunk in the `inner`.
        // After slicing when offset > 0, there may be additional chunks.
        let n_chunks = (len + offset).div_ceil(1024);

        let inner = children.get(0, dtype, len)?;
        let lane_offsets = children.get(1, PType::U32.into(), n_chunks * n_lanes + 1)?;
        let indices = children.get(2, PType::U16.into(), n_patches)?;
        let values = children.get(3, dtype, n_patches)?;

        let data = PatchedData { n_lanes, offset };
        let slots = PatchedSlots {
            inner,
            lane_offsets,
            patch_indices: indices,
            patch_values: values,
        }
        .into_slots();
        let array = Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))?;

        // Like other encodings, the children are not decoded here. Execution checks that every
        // patch index lies within its chunk and clamps lane offsets to the patches, so a corrupt
        // layout cannot write outside the array. `PatchLayout::validate` checks it in full.

        Ok(array.into_array())
    }
}

/// Read the lane offset at `idx`.
fn lane_offset(lane_offsets: &ArrayRef, idx: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    lane_offsets
        .execute_scalar(idx, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("lane offsets must be valid"))
}
