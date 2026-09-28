// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
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
use crate::arrays::Dict;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::DictData;
use crate::arrays::dict::DictMetadata;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

impl ArrayPlugin for Dict {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Dict>()
            .ok_or_else(|| vortex_err!("Dict plugin cannot serialize {}", array.encoding_id()))?;
        let metadata = DictMetadata {
            codes_ptype: PType::try_from(view.codes().dtype())? as i32,
            values_len: u32::try_from(view.values().len()).map_err(|_| {
                vortex_err!(
                    "Dictionary values size {} overflowed u32",
                    view.values().len()
                )
            })?,
            is_nullable_codes: Some(view.codes().dtype().is_nullable()),
            all_values_referenced: Some(view.has_all_values_referenced()),
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
            "Dict plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = DictMetadata::decode(metadata)?;
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, unsafe {
                DictData::new_unchecked().set_all_values_referenced(all_values_referenced)
            })
            .with_slots(smallvec![Some(codes), Some(values)]),
        )?
        .into_array())
    }
}
