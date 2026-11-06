// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Unary`].

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::Unary;
use super::UnaryMetadata;
use super::num_ones_of;
use crate::UnaryData;

impl ArrayPlugin for Unary {
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
            "Unary plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<Unary>();
        let metadata = UnaryMetadata {
            num_ones: num_ones_of(array)?,
            offset: u32::try_from(array.meta.offset())?,
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
            "Unary plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        vortex_ensure_eq!(buffers.len(), 1);
        vortex_ensure_eq!(children.len(), 0);
        let metadata = UnaryMetadata::decode(metadata)?;
        let data = UnaryData::from_buffer(
            buffers[0].clone(),
            metadata.num_ones,
            usize::try_from(metadata.offset)?,
            len,
        )?;
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            ArraySlots::new(),
        ))?
        .into_array())
    }
}
