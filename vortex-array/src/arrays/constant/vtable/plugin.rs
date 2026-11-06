// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Constant`].

use vortex_error::VortexResult;
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
use crate::arrays::Constant;
use crate::arrays::constant::ConstantData;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl ArrayPlugin for Constant {
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
            "Constant plugin cannot serialize an array of another encoding",
        );
        // HACK: Because the scalar is stored in the buffers, we do not need to serialize the
        // metadata at all.
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Constant plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            buffers,
            ..
        } = parts;
        vortex_ensure_eq!(buffers.len(), 1);

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ConstantData::new(scalar),
            ArraySlots::new(),
        ))?
        .into_array())
    }
}
