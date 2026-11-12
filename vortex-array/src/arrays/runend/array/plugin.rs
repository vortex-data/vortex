// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`RunEnd`].

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::RunEnd;
use super::RunEndArrayExt;
use super::RunEndArraySlotsExt;
use super::RunEndData;
use super::RunEndMetadata;
use super::RunEndSlots;
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
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

impl ArrayPlugin for RunEnd {
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
            "RunEnd plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<RunEnd>();
        let metadata = RunEndMetadata {
            ends_ptype: PType::try_from(array.ends().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: array.ends().len() as u64,
            offset: array.offset() as u64,
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
            "RunEnd plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        let metadata = RunEndMetadata::decode(metadata)?;
        let ends_ptype = metadata.get_ends_ptype()?;
        let ends_dtype = DType::Primitive(ends_ptype, Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).map_err(|_| {
            vortex_err!(
                "RunEndMetadata num_runs {} exceeds usize range",
                metadata.num_runs
            )
        })?;
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;
        let offset = usize::try_from(metadata.offset).map_err(|_| {
            vortex_err!(
                "RunEndMetadata offset {} exceeds usize range",
                metadata.offset
            )
        })?;
        RunEndData::validate_parts(
            &ends,
            &values,
            offset,
            len,
            &mut session.create_execution_ctx(),
        )?;
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = RunEndData::new(offset);
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
