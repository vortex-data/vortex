// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementations for [`ALP`].
//!
//! [`ALP`] itself reads and writes the current `vortex.alp` form. [`ALPPatchedPlugin`] reads an
//! ALP node with interior patches as a `PatchedArray` wrapping a patchless ALP array, which keeps
//! previously written datasets readable at no cost.

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ALP;
use super::ALPArrayExt;
use super::ALPArrayOwnedExt;
use super::ALPData;
use super::ALPMetadata;
use crate::alp::Exponents;

impl ArrayPlugin for ALP {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "ALP plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<ALP>();
        let exponents = array.exponents();
        let metadata = ALPMetadata {
            exp_e: exponents.e as u32,
            exp_f: exponents.f as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array.array(),
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "ALP plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        let metadata = ALPMetadata::decode(metadata)?;
        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_bail!("expected type: f32 or f64 but instead got {}", d),
        };
        let encoded = children.get(0, &encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(1, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(2, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(3, &dtype, usize::try_from(p.chunk_offsets_len())?))
                    .transpose()?;

                Patches::from_metadata(
                    len,
                    &p,
                    indices,
                    values,
                    chunk_offsets,
                    &mut session.create_execution_ctx(),
                )
            })
            .transpose()?;

        let slots = ALPData::make_slots(&encoded, patches.as_ref());
        let data = ALPData::new(
            Exponents {
                e: u8::try_from(metadata.exp_e)?,
                f: u8::try_from(metadata.exp_f)?,
            },
            patches,
        );
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))?
        .into_array())
    }
}

/// Custom deserialization plugin that converts an ALP array with interior
/// patches into a PatchedArray holding an ALP array.
#[derive(Debug, Clone)]
pub(crate) struct ALPPatchedPlugin;

impl ArrayPlugin for ALPPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `ALP` ID so that we can take over its
        // deserialization pathway.
        // TODO(joe): dedup method name
        VTable::id(&ALP)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        // Delegate to ALP's metadata serde
        ArrayPlugin::serialize(&ALP, array, session)
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            self.id(),
            "ALP plugin does not recognize serialized ID"
        );
        let alp_array = ArrayPlugin::deserialize(&ALP, parts, session)?
            .try_downcast::<ALP>()
            .map_err(|_| vortex_err!("ALP plugin returned an array that is not vortex.alp"))?;

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
        id == VTable::id(&Patched) || id == VTable::id(&ALP)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
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
        let session = VortexSession::empty();
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

        let serialization = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = ALPPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                ALPPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        )?;

        assert_arrays_eq!(deserialized, parray, &mut SESSION.create_execution_ctx());

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

        let serialization = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = ALPPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                ALPPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        )?;

        assert_arrays_eq!(deserialized, parray, &mut SESSION.create_execution_ctx());

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

        let serialization = SESSION.array_serialize(&array).unwrap().unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        // This panics because PrimitiveArray has no children and ALP requires encoded child.
        let _result = ALPPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                ALPPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        );
    }
}
