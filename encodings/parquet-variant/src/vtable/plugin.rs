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
use vortex_array::EmptyArrayData;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ParquetVariant;
use super::ParquetVariantMetadataProto;
use crate::array::ParquetVariantArraySlotsExt;
use crate::array::ParquetVariantSlots;

impl ArrayPlugin for ParquetVariant {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<ParquetVariant>().ok_or_else(|| {
            vortex_err!(
                "ParquetVariant plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let typed_value_dtype = view
            .typed_value()
            .map(|tv| tv.dtype().try_into())
            .transpose()?;
        let metadata = ParquetVariantMetadataProto {
            has_value: view.value().is_some(),
            typed_value_dtype,
            value_nullable: view.value().is_some_and(|v| v.dtype().is_nullable()),
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
            "ParquetVariant plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "ParquetVariantArray expects 0 buffers, got {}",
            buffers.len()
        );

        let proto = ParquetVariantMetadataProto::decode(metadata)?;
        let typed_value_dtype = match proto.typed_value_dtype.as_ref() {
            Some(dtype) => Some(DType::from_proto(dtype, session)?),
            None => None,
        };

        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let has_typed_value = typed_value_dtype.is_some();
        vortex_ensure!(
            proto.has_value || has_typed_value,
            "At least one of value or typed_value must be present"
        );

        let expected_children = 1 + proto.has_value as usize + has_typed_value as usize;
        vortex_ensure!(
            children.len() == expected_children || children.len() == expected_children + 1,
            "Expected {} or {} children, got {}",
            expected_children,
            expected_children + 1,
            children.len()
        );

        let (validity, mut child_idx) = if children.len() == expected_children {
            (Validity::from(dtype.nullability()), 0)
        } else {
            (Validity::Array(children.get(0, &Validity::DTYPE, len)?), 1)
        };
        let variant_metadata =
            children.get(child_idx, &DType::Binary(Nullability::NonNullable), len)?;
        child_idx += 1;

        let value = if proto.has_value {
            let v = children.get(child_idx, &DType::Binary(proto.value_nullable.into()), len)?;
            child_idx += 1;
            Some(v)
        } else {
            None
        };

        let typed_value = if has_typed_value {
            // typed_value can be any type — primitive, list, struct, etc.
            let dtype = typed_value_dtype
                .ok_or_else(|| vortex_err!("typed_value_dtype missing for typed_value child"))?;
            let tv = children.get(child_idx, &dtype, len)?;
            Some(tv)
        } else {
            None
        };

        let slots = ParquetVariantSlots {
            validity: validity_to_child(&validity, len),
            metadata: variant_metadata,
            value,
            typed_value,
        }
        .into_slots();
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(slots),
        )?
        .into_array())
    }
}
