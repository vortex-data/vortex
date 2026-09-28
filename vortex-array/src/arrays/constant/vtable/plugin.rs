// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
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
        vortex_ensure!(
            array.is::<Constant>(),
            "Constant plugin cannot serialize {}",
            array.encoding_id()
        );
        // HACK: Because the scalar is stored in the buffers, we do not need to serialize the
        // metadata at all.
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata: _,
            buffers,
            children: _,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Constant plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ConstantData::new(scalar),
        ))?
        .into_array())
    }
}
