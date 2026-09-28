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
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::RunEnd;
use super::RunEndArrayExt;
use super::RunEndArraySlotsExt;
use super::RunEndData;
use super::RunEndMetadata;
use super::RunEndSlots;

impl ArrayPlugin for RunEnd {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<RunEnd>()
            .ok_or_else(|| vortex_err!("RunEnd plugin cannot serialize {}", array.encoding_id()))?;
        let metadata = RunEndMetadata {
            ends_ptype: PType::try_from(view.ends().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: view.ends().len() as u64,
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
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "RunEnd plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = RunEndMetadata::decode(metadata)?;
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;
        let offset = usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize");
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = RunEndData::new(offset);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
