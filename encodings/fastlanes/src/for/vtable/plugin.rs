// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::FoR;
use crate::FoRData;
use crate::r#for::array::FoRArrayExt;

impl ArrayPlugin for FoR {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<FoR>()
            .ok_or_else(|| vortex_err!("FoR plugin cannot serialize {}", array.encoding_id()))?;
        // Note that we **only** serialize the optional scalar value (not including the dtype).
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            ScalarValue::to_proto_bytes(view.reference_scalar().value()),
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
            metadata,
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "FoR plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "FoRArray expects 0 buffers, got {}",
            buffers.len()
        );
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let scalar_value = ScalarValue::from_proto_bytes(metadata, dtype, session)?;
        let reference = Scalar::try_new(dtype.clone(), scalar_value)?;
        let encoded = children.get(0, dtype, len)?;
        let slots = smallvec![Some(encoded)];

        let data = FoRData::try_new(reference)?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
