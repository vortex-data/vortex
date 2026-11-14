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
        let deltas_len = array.deltas().len();
        let canonical_deltas_len = (array.offset() + array.len()).next_multiple_of(FL_CHUNK_SIZE);
        vortex_ensure_eq!(
            deltas_len,
            canonical_deltas_len,
            "DeltaArray deltas length must be (offset + len) rounded up to a multiple of \
             {FL_CHUNK_SIZE}",
        );
        let metadata = DeltaMetadata {
            deltas_len: deltas_len as u64,
            offset: u32::try_from(array.offset())?,
            has_validity: array.validity_child().is_some(),
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
        let metadata = DeltaMetadata::decode(metadata)?;
        let required_children = 2 + usize::from(metadata.has_validity);
        vortex_ensure!(
            children.len() >= required_children,
            "DeltaArray expects at least {required_children} children, got {}",
            children.len()
        );
        vortex_ensure!(
            dtype.is_int(),
            "DeltaArray dtype must be an integer, got {dtype}"
        );
        vortex_ensure!(
            !metadata.has_validity || dtype.is_nullable(),
            "DeltaArray validity child requires a nullable dtype, got {dtype}"
        );
        let offset = metadata.offset as usize;
        vortex_ensure!(
            offset < FL_CHUNK_SIZE,
            "DeltaArray offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        let lanes = lane_count(PType::try_from(dtype)?);

        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        let canonical_deltas_len = offset
            .checked_add(len)
            .and_then(|end| end.checked_next_multiple_of(FL_CHUNK_SIZE))
            .ok_or_else(|| vortex_err!("DeltaArray offset + len overflowed usize"))?;
        vortex_ensure_eq!(
            deltas_len,
            canonical_deltas_len,
            "DeltaArray deltas length must be (offset + len) rounded up to a multiple of \
             {FL_CHUNK_SIZE}",
        );
        let bases_len = deltas_len / FL_CHUNK_SIZE * lanes;

        let bases = children.get(0, &dtype.as_nonnullable(), bases_len)?;
        let deltas = children.get(1, &dtype.as_nonnullable(), deltas_len)?;
        let validity_child = metadata
            .has_validity
            .then(|| children.get(2, &Validity::DTYPE, len))
            .transpose()?;

        let data = DeltaData::try_new(offset)?;
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
