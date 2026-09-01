// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Pco;
use crate::array::deserialize_parts;
use crate::array::is_8_bit;
use crate::array::serialized_metadata;

fn pco_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.pco.v2");
    *ID
}

#[derive(Clone, Debug)]
pub(crate) struct PcoPlugin;

impl ArrayPlugin for PcoPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&Pco)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![self.id(), pco_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.encoding_id() == self.id(),
            "Pco plugin cannot serialize in-memory array {}",
            array.encoding_id(),
        );
        let serialized_id = if is_8_bit(array.dtype()) {
            pco_v2_id()
        } else {
            self.id()
        };
        Ok(Some(ArraySerialization::from_array(
            serialized_id,
            array,
            serialized_metadata(array.as_::<Pco>()),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parts.serialized_id == self.id() || parts.serialized_id == pco_v2_id(),
            "Pco plugin does not recognize serialized ID {}",
            parts.serialized_id,
        );
        vortex_ensure!(
            parts.serialized_id != self.id() || !is_8_bit(parts.dtype),
            "serialized ID vortex.pco cannot represent {}",
            parts.dtype,
        );
        Ok(Array::<Pco>::try_from_parts(deserialize_parts(
            parts.dtype,
            parts.len,
            parts.metadata,
            parts.buffers,
            parts.children,
        )?)?
        .into_array())
    }
}
