// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::dtype::PType;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::Delta;
use super::DeltaMetadata;
use crate::DeltaData;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::DeltaSlots;
use crate::delta::array::lane_count;

impl ArrayPlugin for Delta {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Delta>()
            .ok_or_else(|| vortex_err!("Delta plugin cannot serialize {}", array.encoding_id()))?;
        let metadata = DeltaMetadata {
            deltas_len: view.deltas().len() as u64,
            offset: view.offset() as u32,
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
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Delta plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "DeltaArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 2,
            "DeltaArray expects 2 children, got {}",
            children.len()
        );
        let metadata = DeltaMetadata::decode(metadata)?;
        let ptype = PType::try_from(dtype)?;
        let lanes = lane_count(ptype);

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        let data = DeltaData::try_new(metadata.offset as usize)?;
        let slots = DeltaSlots { bases, deltas }.into_slots();
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
