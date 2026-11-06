// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`VarBin`].

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::VarBinMetadata;
use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
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
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "VarBin plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<VarBin>();
        let metadata = VarBinMetadata {
            offsets_ptype: PType::try_from(array.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
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
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "VarBin plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        let metadata = VarBinMetadata::decode(metadata)?;
        let offsets_ptype = metadata.get_offsets_ptype()?;
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
            &DType::Primitive(offsets_ptype, Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone().try_to_host_sync()?;

        let data = VarBinData::try_build(
            offsets.clone(),
            bytes,
            dtype.clone(),
            validity.clone(),
            &mut session.create_execution_ctx(),
        )?;
        let slots = VarBinData::make_slots(offsets, &validity, len);
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
