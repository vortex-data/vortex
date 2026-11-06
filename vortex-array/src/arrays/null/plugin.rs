// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Null`].

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ArraySlots;
use crate::IntoArray;
use crate::VTable;
use crate::array::EmptyArrayData;
use crate::arrays::Null;

impl ArrayPlugin for Null {
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
            "Null plugin cannot serialize an array of another encoding",
        );
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
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
            "Null plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            ..
        } = parts;
        vortex_ensure!(
            metadata.is_empty(),
            "NullArray expects empty metadata, got {} bytes",
            metadata.len()
        );
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            EmptyArrayData,
            ArraySlots::new(),
        ))?
        .into_array())
    }
}
