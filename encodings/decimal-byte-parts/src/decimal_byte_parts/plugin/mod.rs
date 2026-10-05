// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ArrayPlugin implementation for DBP that handles different wire formats.

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
use vortex_session::registry::CachedId;

use super::DecimalByteParts;
use super::DecimalBytePartsArraySlotsExt;

#[cfg(test)]
mod tests;

mod v1;
mod v2;

pub use v2::DecimalBytePartsV2Metadata;

/// The frozen single-child DBP serialized ID.
pub fn decimal_byte_parts_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
    *ID
}

/// The current in-memory DBP ID and serialized ID for arrays with lower parts.
pub fn decimal_byte_parts_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts.v2");
    *ID
}

/// Serde for the [`DecimalByteParts`] array using the frozen v1 and v2 wire formats.
///
/// Each version owns its metadata schema and serde functions. The plugin writes v1 whenever an
/// array has no lower parts, so such arrays stay readable by older readers, and v2 otherwise.
/// The v2 format itself accepts any lower part count up to the maximum.
///
/// Register this plugin, or call [`crate::initialize`], to enable both formats. Direct registration
/// of [`DecimalByteParts`] does not support serde.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsPlugin;

impl ArrayPlugin for DecimalBytePartsPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&DecimalByteParts)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<DecimalByteParts>().ok_or_else(|| {
            vortex_err!(
                "DecimalByteParts plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let serialized = if view.lower_parts().is_empty() {
            v1::serialize(view)?
        } else {
            v2::serialize(view)?
        };
        Ok(Some(serialized))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let array = if parts.serialized_id == decimal_byte_parts_v1_id() {
            v1::deserialize(parts)?
        } else if parts.serialized_id == decimal_byte_parts_v2_id() {
            v2::deserialize(parts)?
        } else {
            vortex_bail!(
                "DecimalByteParts plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        };
        Ok(array.into_array())
    }
}
