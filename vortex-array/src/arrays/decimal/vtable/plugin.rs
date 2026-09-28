// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::DecimalMetadata;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::arrays::Decimal;
use crate::arrays::decimal::DecimalData;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;

impl ArrayPlugin for Decimal {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<Decimal>().ok_or_else(|| {
            vortex_err!("Decimal plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = DecimalMetadata {
            values_type: view.values_type() as i32,
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
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Decimal plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = DecimalMetadata::decode(metadata)?;
        let values = fixed_width::single_buffer(buffers)?;

        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;

        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Expected Decimal dtype, got {:?}", dtype)
        };

        let slots = DecimalData::make_slots(&validity, len);
        let data = match_each_decimal_value_type!(metadata.values_type(), |D| {
            // Check and reinterpret-cast the buffer
            vortex_ensure!(
                values.is_aligned_to(Alignment::of::<D>()),
                "DecimalArray buffer not aligned for values type {:?}",
                D::DECIMAL_TYPE
            );
            DecimalData::try_new_handle(values, metadata.values_type(), *decimal_dtype)
        })?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
