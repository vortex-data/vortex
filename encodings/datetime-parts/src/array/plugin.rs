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
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::DateTimeParts;
use super::DateTimePartsArraySlotsExt;
use super::DateTimePartsData;
use super::DateTimePartsMetadata;

impl ArrayPlugin for DateTimeParts {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<DateTimeParts>().ok_or_else(|| {
            vortex_err!(
                "DateTimeParts plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let metadata = DateTimePartsMetadata {
            days_ptype: PType::try_from(view.days().dtype())? as i32,
            seconds_ptype: PType::try_from(view.seconds().dtype())? as i32,
            subseconds_ptype: PType::try_from(view.subseconds().dtype())? as i32,
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
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "DateTimeParts plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = DateTimePartsMetadata::decode(metadata)?;
        if children.len() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                children.len()
            )
        }

        let days = children.get(
            0,
            &DType::Primitive(metadata.get_days_ptype()?, dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            1,
            &DType::Primitive(metadata.get_seconds_ptype()?, Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(metadata.get_subseconds_ptype()?, Nullability::NonNullable),
            len,
        )?;

        let slots = smallvec![Some(days), Some(seconds), Some(subseconds)];
        let data = DateTimePartsData {};
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
