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
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::RLE;
use super::RLEMetadata;
use crate::RLEData;
use crate::rle::array::RLEArrayExt;
use crate::rle::array::RLEArraySlotsExt;
use crate::rle::array::RLESlots;

impl ArrayPlugin for RLE {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<RLE>()
            .ok_or_else(|| vortex_err!("RLE plugin cannot serialize {}", array.encoding_id()))?;
        let metadata = RLEMetadata {
            values_len: view.values().len() as u64,
            indices_len: view.indices().len() as u64,
            indices_ptype: PType::try_from(view.indices().dtype())? as i32,
            values_idx_offsets_len: view.values_idx_offsets().len() as u64,
            values_idx_offsets_ptype: PType::try_from(view.values_idx_offsets().dtype())? as i32,
            offset: view.offset() as u64,
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
            "RLE plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "RLEArray expects 0 buffers, got {}",
            buffers.len()
        );
        let metadata = RLEMetadata::decode(metadata)?;
        let values = children.get(
            0,
            &DType::Primitive(dtype.as_ptype(), Nullability::NonNullable),
            usize::try_from(metadata.values_len)?,
        )?;

        let indices = children.get(
            1,
            &DType::Primitive(metadata.indices_ptype(), dtype.nullability()),
            usize::try_from(metadata.indices_len)?,
        )?;

        let values_idx_offsets = children.get(
            2,
            &DType::Primitive(
                metadata.values_idx_offsets_ptype(),
                Nullability::NonNullable,
            ),
            usize::try_from(metadata.values_idx_offsets_len)?,
        )?;

        let slots = RLESlots {
            values,
            indices,
            values_idx_offsets,
        }
        .into_slots();
        let data = RLEData::try_new(metadata.offset as usize)?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
