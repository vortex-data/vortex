// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::VarBinMetadata;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbin::VarBinData;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

impl ArrayPlugin for VarBin {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<VarBin>()
            .ok_or_else(|| vortex_err!("VarBin plugin cannot serialize {}", array.encoding_id()))?;
        let metadata = VarBinMetadata {
            offsets_ptype: PType::try_from(view.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
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
            "VarBin plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = VarBinMetadata::decode(metadata)?;
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        let offsets = children.get(
            0,
            &DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone().try_to_host_sync()?;

        let data = VarBinData::try_build(offsets.clone(), bytes, dtype.clone(), validity.clone())?;
        let slots = VarBinData::make_slots(offsets, &validity, len);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
