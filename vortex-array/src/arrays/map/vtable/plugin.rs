// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Map`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::arrays::ListView;
use crate::arrays::Map;
use crate::arrays::map::MapData;
use crate::arrays::map::MapSlots;
use crate::dtype::DType;

impl ArrayPlugin for Map {
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
            "Map plugin cannot serialize an array of another encoding",
        );
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Map plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        if !metadata.is_empty() {
            vortex_bail!(
                "MapArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        vortex_ensure!(buffers.is_empty(), "MapArray expects no buffers");

        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        vortex_ensure_eq!(children.len(), MapSlots::COUNT);

        let expected_entries_dtype =
            DType::List(std::sync::Arc::new(map_dtype.entries_dtype()), *nullability);
        let entries = children.get(MapSlots::ENTRIES, &expected_entries_dtype, len)?;
        vortex_ensure!(
            entries.is::<ListView>(),
            "MapArray entries must use vortex.listview encoding, got {}",
            entries.encoding_id()
        );

        let slots = MapData::make_slots(entries);
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            MapData,
            slots,
        ))?
        .into_array())
    }
}
