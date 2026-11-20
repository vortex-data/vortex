// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementations for `BitPacked`.
//!
//! [`BitPackedPlugin`] owns serde for `fastlanes.bitpacked` and `fastlanes.bitpacked.v2`.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;

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
/// Register this plugin, or call [`crate::initialize`], to enable serde. [`BitPacked`] does not
/// implement [`ArrayPlugin`], so it cannot be registered on its own.
#[derive(Clone, Debug)]
pub struct BitPackedPlugin;

impl ArrayPlugin for BitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&BitPacked)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![bitpacked_v1_id(), bitpacked_v2_id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
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
            BitWidthsView::Global(bit_width) => v1::serialize(array, view, bit_width)?,
            BitWidthsView::Blocked(block_offsets) => v2::serialize(array, view, block_offsets)?,
        };
        Ok(Some(serialization))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == bitpacked_v1_id() {
            v1::deserialize(parts, session)
        } else if parts.serialized_id == bitpacked_v2_id() {
            v2::deserialize(parts, session)
        } else {
            vortex_bail!(
                "BitPacked plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        }
    }
}
