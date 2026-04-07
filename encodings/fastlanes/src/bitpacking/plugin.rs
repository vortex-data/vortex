// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A custom [`ArrayPlugin`] that lets you load in and deserialize a `BitPacked` array with interior
//! patches as a `PatchedArray` that wraps a patchless `BitPacked` array.
//!
//! This enables zero-cost backward compatibility with previously written datasets.

use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArrayExt;

/// Custom deserialization plugin that converts a BitPacked array with interior
/// Patches into a PatchedArray holding a BitPacked array.
#[derive(Debug, Clone)]
pub struct BitPackedPatchedPlugin;

impl ArrayPlugin for BitPackedPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `BitPacked` ID so that we can take over its
        // deserialization pathway.
        BitPacked::ID
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // delegate to BitPacked VTable for serialization
        BitPacked.serialize(array, session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let bitpacked = BitPacked
            .deserialize(dtype, len, metadata, buffers, children, session)?
            .try_downcast::<BitPacked>()
            .map_err(|_| {
                vortex_err!("BitPacked plugin should only deserialize vortex.bitpacked")
            })?;

        // Create a new BitPackedArray without the interior patches installed.
        let Some(patches) = bitpacked.patches() else {
            return Ok(bitpacked.into_array());
        };

        let packed = bitpacked.packed().clone();
        let ptype = bitpacked.dtype().as_ptype();
        let validity = bitpacked.validity()?;
        let bw = bitpacked.bit_width;
        let len = bitpacked.len();
        let offset = bitpacked.offset();

        let bitpacked_without_patches =
            BitPacked::try_new(packed, ptype, validity, None, bw, len, offset)?.into_array();

        let patched = Patched::from_array_and_patches(
            bitpacked_without_patches,
            &patches,
            &mut session.create_execution_ctx(),
        )?;

        Ok(patched.into_array())
    }
}
