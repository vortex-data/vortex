// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Patched`].

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::PatchedMetadata;
use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::arrays::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::dtype::PType;

impl ArrayPlugin for Patched {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "Patched plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<Patched>();
        // Readers derive the lane offsets length from the metadata, so reject arrays they could
        // not read back.
        let data = PatchedData {
            n_lanes: array.n_lanes(),
            offset: array.offset(),
        };
        vortex_ensure_eq!(
            array.lane_offsets().len(),
            data.lane_offsets_len(array.dtype(), array.len())?,
        );
        let metadata = PatchedMetadata {
            n_patches: u32::try_from(array.patch_indices().len())?,
            n_lanes: u32::try_from(array.n_lanes())?,
            offset: u32::try_from(array.offset())?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array.array(),
            metadata,
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
            buffers,
            children,
            ..
        } = parts;
        vortex_ensure!(
            buffers.is_empty(),
            "PatchedArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() >= PatchedSlots::COUNT,
            "PatchedArray expects at least {} children, got {}",
            PatchedSlots::COUNT,
            children.len()
        );
        let metadata = PatchedMetadata::decode(metadata)?;
        let n_patches = usize::try_from(metadata.n_patches)?;
        let data = PatchedData {
            n_lanes: usize::try_from(metadata.n_lanes)?,
            offset: usize::try_from(metadata.offset)?,
        };
        // Check the metadata before reading children, whose lengths and dtypes derive from it.
        let lane_offsets_len = data.lane_offsets_len(dtype, len)?;

        let inner = children.get(PatchedSlots::INNER, dtype, len)?;
        let lane_offsets = children.get(
            PatchedSlots::LANE_OFFSETS,
            PType::U32.into(),
            lane_offsets_len,
        )?;
        let indices = children.get(PatchedSlots::PATCH_INDICES, PType::U16.into(), n_patches)?;
        let values = children.get(
            PatchedSlots::PATCH_VALUES,
            &dtype.as_nonnullable(),
            n_patches,
        )?;

        let slots = PatchedSlots {
            inner,
            lane_offsets,
            patch_indices: indices,
            patch_values: values,
        }
        .into_slots();
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))?
        .into_array())
    }
}
