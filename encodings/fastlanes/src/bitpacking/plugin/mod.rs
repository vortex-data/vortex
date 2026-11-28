// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementations for `BitPacked`.
//!
//! [`BitPackedPlugin`] owns serde for `fastlanes.bitpacked`.
//! [`BitPackedPatchedPlugin`] lets you load in and deserialize a `BitPacked` array with interior
//! patches as a `PatchedArray` that wraps a patchless `BitPacked` array.
//!
//! This enables zero-cost backward compatibility with previously written datasets.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;

#[cfg(test)]
mod tests;
mod v1;

/// Serde for the [`BitPacked`] array (`fastlanes.bitpacked`).
///
/// Register this plugin, or call [`crate::initialize`], to enable serde. [`BitPacked`] does not
/// implement [`ArrayPlugin`], so it cannot be registered on its own.
#[derive(Clone, Debug)]
pub struct BitPackedPlugin;

impl ArrayPlugin for BitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&BitPacked)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<BitPacked>().ok_or_else(|| {
            vortex_err!("BitPacked plugin cannot serialize {}", array.encoding_id())
        })?;
        Ok(Some(v1::serialize(array, view, view.bit_width())?))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayVTable::id(&BitPacked),
            "BitPacked plugin does not recognize serialized ID {}",
            parts.serialized_id
        );
        v1::deserialize(parts, session)
    }
}

/// Custom deserialization plugin that converts a BitPacked array with interior
/// Patches into a PatchedArray holding a BitPacked array.
#[derive(Debug, Clone)]
pub(crate) struct BitPackedPatchedPlugin;

impl ArrayPlugin for BitPackedPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `BitPacked` ID so that we can take over its
        // deserialization pathway.
        // TODO(joe): dedup method name
        ArrayVTable::id(&BitPacked)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        // delegate to BitPackedPlugin for serialization
        BitPackedPlugin.serialize(array, session)
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayVTable::id(&BitPacked),
            "BitPacked plugin does not recognize serialized ID {}",
            parts.serialized_id,
        );
        let bitpacked: BitPackedArray = BitPackedPlugin
            .deserialize(parts, session)?
            .try_downcast()
            .map_err(|_| {
                vortex_err!("BitPacked plugin should only deserialize fastlanes.bitpacked")
            })?;

        // Create a new BitPackedArray without the interior patches installed.
        let Some(patches) = bitpacked.patches() else {
            return Ok(bitpacked.into_array());
        };

        let packed = bitpacked.packed().clone();
        let ptype = bitpacked.dtype().as_ptype();
        let validity = bitpacked.validity()?;
        let len = bitpacked.len();
        let offset = bitpacked.offset();
        let bit_width = bitpacked.bit_width();

        let bitpacked_without_patches =
            BitPacked::try_new(packed, ptype, validity, None, bit_width, len, offset)?.into_array();

        let patched = Patched::from_array_and_patches(
            bitpacked_without_patches,
            &patches,
            &mut session.create_execution_ctx(),
        )?;

        Ok(patched.into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == ArrayVTable::id(&BitPacked) || id == ArrayVTable::id(&Patched)
    }
}
