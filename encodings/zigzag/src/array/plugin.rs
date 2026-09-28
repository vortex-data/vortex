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
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use super::ZigZag;
use super::ZigZagData;

impl ArrayPlugin for ZigZag {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.is::<ZigZag>(),
            "ZigZag plugin cannot serialize {}",
            array.encoding_id()
        );
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
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
            "ZigZag plugin does not recognize serialized ID {serialized_id}"
        );
        if !metadata.is_empty() {
            vortex_bail!(
                "ZigZagArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        let slots = smallvec![Some(encoded.clone())];
        let data = ZigZagData::try_new(encoded.dtype())?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
