// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::VariantMetadataProto;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::arrays::Variant;
use crate::arrays::variant::VariantSlots;
use crate::dtype::DType;

impl ArrayPlugin for Variant {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<Variant>().ok_or_else(|| {
            vortex_err!("Variant plugin cannot serialize {}", array.encoding_id())
        })?;
        let shredded_dtype = view.slots()[VariantSlots::SHREDDED]
            .as_ref()
            .map(|shredded| shredded.dtype().try_into())
            .transpose()?;
        let metadata = VariantMetadataProto { shredded_dtype }.encode_to_vec();
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
            "Variant plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "VariantArray expects 0 buffers, got {}",
            buffers.len()
        );
        let proto = VariantMetadataProto::decode(metadata)?;
        let shredded_dtype = proto
            .shredded_dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?;
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let expected_children = 1 + usize::from(shredded_dtype.is_some());
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {} children, got {}",
            expected_children,
            children.len(),
        );
        let core_storage = children.get(0, dtype, len)?;
        let shredded = shredded_dtype
            .map(|dtype| children.get(1, &dtype, len))
            .transpose()?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(
                VariantSlots {
                    core_storage,
                    shredded,
                }
                .into_slots(),
            ),
        )?
        .into_array())
    }
}
