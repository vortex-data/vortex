// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ArrayPlugin implementation for FoR.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

#[cfg(test)]
mod tests;
mod v1;
mod v2;

/// The frozen FoR serialized ID for arrays with a single reference.
pub fn for_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.for");
    *ID
}

/// The in-memory FoR ID, and the serialized ID for arrays whose chunks have different references.
pub fn for_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.for.v2");
    *ID
}

/// Serde for the [`FoR`] array.
///
/// Arrays with constant references serialize as `fastlanes.for`, which stores the reference in
/// its metadata. Arrays whose chunks have different references serialize as
/// `fastlanes.for.v2`, which stores them as a child.
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
        vec![for_v1_id(), for_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<FoR>()
            .ok_or_else(|| vortex_err!("FoR plugin cannot serialize {}", array.encoding_id()))?;
        Ok(Some(match view.constant_reference() {
            Some(reference) => v1::serialize(view, &reference),
            None => v2::serialize(view),
        }))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == for_v1_id() {
            v1::deserialize(parts, session)
        } else if parts.serialized_id == for_v2_id() {
            v2::deserialize(parts)
        } else {
            vortex_bail!(
                "FoR plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        }
    }
}
