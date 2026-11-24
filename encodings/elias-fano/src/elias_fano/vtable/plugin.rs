// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`EliasFano`].

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
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::EliasFano;
use super::EliasFanoMetadata;
use super::UPPER_DTYPE;
use crate::EliasFanoData;
use crate::elias_fano::array::EliasFanoArraySlotsExt;
use crate::elias_fano::array::EliasFanoSlots;

impl ArrayPlugin for EliasFano {
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
            "EliasFano plugin cannot serialize an array of another encoding",
        );

        let array = array.as_::<EliasFano>();
        let metadata = EliasFanoMetadata {
            lower_bits: u32::from(array.lower_bits),
            num_buckets: u64::try_from(array.upper().len())?,
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
            "EliasFano plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        vortex_ensure_eq!(buffers.len(), 0);
        vortex_ensure_eq!(children.len(), EliasFanoSlots::COUNT);

        let metadata = EliasFanoMetadata::decode(metadata)?;
        let upper = children.get(
            EliasFanoSlots::UPPER,
            &UPPER_DTYPE,
            usize::try_from(metadata.num_buckets)?,
        )?;
        let lower = children.get(EliasFanoSlots::LOWER, dtype, len)?;
        let data = EliasFanoData {
            lower_bits: u8::try_from(metadata.lower_bits)?,
        };

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            EliasFanoSlots { upper, lower }.into_slots(),
        ))?
        .into_array())
    }
}
