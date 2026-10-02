// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementations for `BitPacked`.
//!
//! [`BitPackedPlugin`] owns serde for `fastlanes.bitpacked` and `fastlanes.bitpacked.v2`.
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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitWidths;

#[cfg(test)]
mod tests;
mod v1;
mod v2;

/// The frozen BitPacked serialized ID for arrays whose blocks share one bit width.
pub fn bitpacked_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.bitpacked");
    *ID
}

/// The in-memory BitPacked ID, and the serialized ID for arrays whose blocks have their own bit
/// widths.
pub fn bitpacked_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.bitpacked.v2");
    *ID
}

/// Serde for the [`BitPacked`] array.
///
/// Arrays with a global bit width serialize as `fastlanes.bitpacked`, which stores the width in
/// its metadata. Arrays whose blocks have their own bit widths serialize as
/// `fastlanes.bitpacked.v2`, which stores the block offsets as a child.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde. Direct registration of
/// [`BitPacked`] does not support serde.
#[derive(Clone, Debug)]
pub struct BitPackedPlugin;

impl ArrayPlugin for BitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&BitPacked)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![bitpacked_v1_id(), bitpacked_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<BitPacked>().ok_or_else(|| {
            vortex_err!("BitPacked plugin cannot serialize {}", array.encoding_id())
        })?;
        let serialization = match view.bit_widths() {
            BitWidths::Global(bit_width) => v1::serialize(array, view, bit_width)?,
            BitWidths::Blocked(block_offsets) => v2::serialize(array, view, &block_offsets)?,
        };
        Ok(Some(serialization))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == bitpacked_v1_id() {
            v1::deserialize(parts)
        } else if parts.serialized_id == bitpacked_v2_id() {
            v2::deserialize(parts)
        } else {
            vortex_bail!(
                "BitPacked plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        }
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

    fn serialized_ids(&self) -> Vec<ArrayId> {
        BitPackedPlugin.serialized_ids()
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
        vortex_ensure!(
            self.serialized_ids().contains(&parts.serialized_id),
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

        let bitpacked_without_patches = match bitpacked.bit_widths() {
            BitWidths::Global(bw) => {
                BitPacked::try_new(packed, ptype, validity, None, bw, len, offset)?
            }
            BitWidths::Blocked(block_offsets) => BitPacked::try_new_with_block_offsets(
                packed,
                ptype,
                validity,
                None,
                block_offsets,
                len,
                offset,
            )?,
        }
        .into_array();

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
