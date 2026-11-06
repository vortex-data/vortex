// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`DateTimeParts`].

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
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
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
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "DateTimeParts plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<DateTimeParts>();
        let metadata = DateTimePartsMetadata {
            days_ptype: PType::try_from(array.days().dtype())? as i32,
            seconds_ptype: PType::try_from(array.seconds().dtype())? as i32,
            subseconds_ptype: PType::try_from(array.subseconds().dtype())? as i32,
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
            "DateTimeParts plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
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
