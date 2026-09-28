// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayDeserialization;
use vortex::array::ArrayId;
use vortex::array::ArrayPlugin;
use vortex::array::ArrayRef;
use vortex::array::ArraySerialization;
use vortex::array::VTable;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::session::VortexSession;

use super::PythonVTable;

impl ArrayPlugin for PythonVTable {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        _array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("PythonArray deserialization is not supported");
    }
}
