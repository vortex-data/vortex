// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ArrayPlugin implementation for FoR that handles its wire formats.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::FoR;

#[cfg(test)]
mod tests;

mod v1;

/// The frozen single-reference FoR serialized ID.
pub fn for_v1_id() -> ArrayId {
    VTable::id(&FoR)
}

/// Serde for the [`FoR`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde. Direct registration of
/// [`FoR`] does not support serde.
#[derive(Clone, Debug)]
pub struct FoRPlugin;

impl ArrayPlugin for FoRPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&FoR)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![for_v1_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<FoR>()
            .ok_or_else(|| vortex_err!("FoR plugin cannot serialize {}", array.encoding_id()))?;
        Ok(Some(v1::serialize(view)))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == for_v1_id() {
            Ok(v1::deserialize(parts, session)?.into_array())
        } else {
            vortex_bail!(
                "FoR plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        }
    }
}
