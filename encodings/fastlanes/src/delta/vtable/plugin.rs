// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Delta`].

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::Delta;
use super::DeltaMetadata;
use crate::DeltaData;
use crate::FL_CHUNK_SIZE;
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
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "Delta plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<Delta>();
        let metadata = DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
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
            "Delta plugin does not recognize the serialized ID",
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
            "DeltaArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 2 || children.len() == 3,
            "DeltaArray expects 2 or 3 children, got {}",
            children.len()
        );
        let metadata = DeltaMetadata::decode(metadata)?;
        let ptype = PType::try_from(dtype)?;
        let lanes = lane_count(ptype);

        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        vortex_ensure!(
            deltas_len.is_multiple_of(FL_CHUNK_SIZE),
            "deltas length must be a multiple of {FL_CHUNK_SIZE}"
        );
        let bases_len = deltas_len / FL_CHUNK_SIZE * lanes;

        let bases = children.get(0, &dtype.as_nonnullable(), bases_len)?;
        let deltas = children.get(1, &dtype.as_nonnullable(), deltas_len)?;
        let validity_child = if children.len() == 3 {
            Some(children.get(2, &Validity::DTYPE, len)?)
        } else {
            None
        };

        let data = DeltaData::try_new(metadata.offset as usize)?;
        let slots = DeltaSlots {
            bases,
            deltas,
            validity_child,
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
