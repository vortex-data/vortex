// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::PatchedMetadata;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
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
        let view = array.as_opt::<Patched>().ok_or_else(|| {
            vortex_err!("Patched plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = PatchedMetadata {
            n_patches: u32::try_from(view.patch_indices().len())?,
            n_lanes: u32::try_from(view.n_lanes())?,
            offset: u32::try_from(view.offset())?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata,
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Patched plugin does not recognize serialized ID {serialized_id}"
        );
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
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
