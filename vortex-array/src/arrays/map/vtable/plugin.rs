// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
        vortex_ensure!(
            array.is::<Map>(),
            "Map plugin cannot serialize {}",
            array.encoding_id()
        );
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
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
            "Map plugin does not recognize serialized ID {serialized_id}"
        );
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
        vortex_ensure!(
            children.len() == MapSlots::COUNT,
            "MapArray expected {} child, found {}",
            MapSlots::COUNT,
            children.len()
        );

        let expected_entries_dtype =
            DType::List(std::sync::Arc::new(map_dtype.entries_dtype()), *nullability);
        let entries = children.get(MapSlots::ENTRIES, &expected_entries_dtype, len)?;
        vortex_ensure!(
            entries.is::<ListView>(),
            "MapArray entries must use vortex.listview encoding, got {}",
            entries.encoding_id()
        );

        let slots = MapData::make_slots(entries);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, MapData).with_slots(slots),
        )?
        .into_array())
    }
}
