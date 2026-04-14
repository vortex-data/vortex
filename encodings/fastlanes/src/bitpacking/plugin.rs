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
pub(crate) struct BitPackedPatchedPlugin;

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
                vortex_err!("BitPacked plugin should only deserialize fastlanes.bitpacked")
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

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == &BitPacked::ID || id == &Patched.id()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::BitPackedPatchedPlugin;
    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(BitPackedPatchedPlugin);
        session
    });

    #[test]
    fn test_decode_bitpacked_patches() -> VortexResult<()> {
        // Create values where some exceed the bit width, causing patches.
        // With bit_width=9, max value is 511. Values >=512 become patches.
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.into_array();
        let bitpacked = BitPackedData::encode(&parray, 9)?;

        assert!(
            bitpacked.patches().is_some(),
            "Expected BitPacked array to have patches"
        );

        let array = bitpacked.as_array();

        let metadata = array.metadata(&SESSION)?.unwrap_or_default();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = BitPackedPatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        )?;

        let patched: PatchedArray = deserialized
            .try_downcast()
            .map_err(|a| vortex_err!("Expected Patched, got {}", a.encoding_id()))?;

        let inner_bitpacked: BitPackedArray = patched
            .inner()
            .clone()
            .try_downcast()
            .map_err(|a| vortex_err!("Expected inner BitPacked, got {}", a.encoding_id()))?;

        assert!(
            inner_bitpacked.patches().is_none(),
            "Inner BitPacked should NOT have patches"
        );

        Ok(())
    }

    #[test]
    fn bitpacked_without_patches_stays_bitpacked() -> VortexResult<()> {
        // With bit_width=16, max value is 65535. All values 0..100 fit.
        let values: Buffer<i32> = (0i32..100).collect();
        let parray = values.into_array();
        let bitpacked = BitPackedData::encode(&parray, 16)?;

        assert!(
            bitpacked.patches().is_none(),
            "Expected BitPacked array without patches"
        );

        let array = bitpacked.as_array();

        let metadata = array.metadata(&SESSION)?.unwrap_or_default();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = BitPackedPatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        )?;

        let result = deserialized
            .try_downcast::<BitPacked>()
            .map_err(|a| vortex_err!("Expected deserialize BitPacked, got {}", a.encoding_id()))?;

        assert!(result.patches().is_none(), "Result should not have patches");

        Ok(())
    }

    #[test]
    fn primitive_array_returns_error() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

        let metadata = array.metadata(&SESSION)?.unwrap_or_default();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let result = BitPackedPatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        );

        assert!(
            result.is_err(),
            "Expected error when deserializing PrimitiveArray with BitPackedPatchedPlugin"
        );

        Ok(())
    }
}
