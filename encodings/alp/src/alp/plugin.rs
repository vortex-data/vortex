// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A custom [`ArrayPlugin`] that lets you load in and deserialize an `ALP` array with interior
//! patches as a `PatchedArray` that wraps a patchless `ALP` array.
//!
//! This enables zero-cost backward compatibility with previously written datasets.

use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArrayOwnedExt;

/// Custom deserialization plugin that converts an ALP array with interior
/// patches into a PatchedArray holding an ALP array.
#[derive(Debug, Clone)]
pub(crate) struct ALPPatchedPlugin;

impl ArrayPlugin for ALPPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `ALP` ID so that we can take over its
        // deserialization pathway.
        // TODO(joe): dedup method name
        ArrayVTable::id(&ALP)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Delegate to ALP's metadata serde
        ALP.serialize(array, session)
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
        let alp_array = Array::<ALP>::try_from_parts(ArrayVTable::deserialize(
            &ALP, dtype, len, metadata, buffers, children, session,
        )?)
        .map_err(|_| vortex_err!("ALP plugin should only deserialize vortex.alp"))?;

        // Check if there are interior patches to externalize.
        let Some(patches) = alp_array.patches() else {
            return Ok(alp_array.into_array());
        };

        // Extract components and create a new ALP array without patches.
        let (encoded, exponents, _) = alp_array.into_parts();

        let alp_without_patches = ALP::try_new(encoded, exponents, None)?.into_array();

        let patched = Patched::from_array_and_patches(
            alp_without_patches,
            &patches,
            &mut session.create_execution_ctx(),
        )?;

        Ok(patched.into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        // TODO(joe): dedup method name
        id == ArrayVTable::id(&Patched) || id == ArrayVTable::id(&ALP)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::ALPPatchedPlugin;
    use crate::ALP;
    use crate::ALPArray;
    use crate::ALPArrayExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(ALPPatchedPlugin);
        session
    });

    #[test]
    fn test_decode_alp_patches() -> VortexResult<()> {
        // Create values where some don't encode cleanly with ALP, causing patches.
        // PI doesn't encode cleanly.
        let values: Vec<f64> = (0..100)
            .map(|i| if i % 4 == 3 { PI } else { i as f64 })
            .collect();

        let parray = PrimitiveArray::from_iter(values);
        let alp_encoded = alp_encode(parray.as_view(), None, &mut SESSION.create_execution_ctx())?;

        assert!(
            alp_encoded.patches().is_some(),
            "Expected ALP array to have patches"
        );

        let array = alp_encoded.as_array();

        let metadata = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = ALPPatchedPlugin.deserialize(
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

        let inner_alp: ALPArray = patched
            .inner()
            .clone()
            .try_downcast()
            .map_err(|a| vortex_err!("Expected inner ALP, got {}", a.encoding_id()))?;

        assert!(
            inner_alp.patches().is_none(),
            "Inner ALP should NOT have patches"
        );

        Ok(())
    }

    #[test]
    fn alp_without_patches_stays_alp() -> VortexResult<()> {
        // Values that encode cleanly without patches.
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let parray = PrimitiveArray::from_iter(values);
        let alp_encoded = alp_encode(parray.as_view(), None, &mut SESSION.create_execution_ctx())?;

        assert!(
            alp_encoded.patches().is_none(),
            "Expected ALP array without patches"
        );

        let array = alp_encoded.as_array();

        let metadata = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = ALPPatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        )?;

        let result = deserialized
            .try_downcast::<ALP>()
            .map_err(|a| vortex_err!("Expected deserialized ALP, got {}", a.encoding_id()))?;

        assert!(result.patches().is_none(), "Result should not have patches");

        Ok(())
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn primitive_array_returns_error() {
        let array = PrimitiveArray::from_iter([1.0f64, 2.0, 3.0]).into_array();

        let metadata = SESSION.array_serialize(&array).unwrap().unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        // This panics because PrimitiveArray has no children and ALP requires encoded child.
        let _result = ALPPatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        );
    }
}
