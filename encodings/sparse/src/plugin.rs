// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Sparse`].

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
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Sparse;
use crate::SparseData;
use crate::SparseExt;
use crate::SparseMetadata;

impl ArrayPlugin for Sparse {
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
            "Sparse plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<Sparse>();
        let patches = array.patches().to_metadata(array.len(), array.dtype())?;
        let metadata = SparseMetadata { patches };

        // Note that we DO NOT serialize the fill value since that is stored in the buffers.
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array.array(),
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        deserialize(parts, session, use_experimental_patches())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == VTable::id(&Sparse) || (use_experimental_patches() && id == VTable::id(&Patched))
    }
}

fn deserialize(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
    externalize: bool,
) -> VortexResult<ArrayRef> {
    vortex_ensure_eq!(
        parts.serialized_id,
        VTable::id(&Sparse),
        "Sparse plugin does not recognize the serialized ID",
    );
    let ArrayDeserialization {
        dtype,
        len,
        metadata,
        buffers,
        children,
        ..
    } = parts;
    let metadata = SparseMetadata::decode(metadata)?;

    // Once we have the patches metadata, we need to get the fill value from the buffers.

    if buffers.len() != 1 {
        vortex_bail!("Expected 1 buffer, got {}", buffers.len());
    }
    let scalar_bytes: &[u8] = &buffers[0].clone().try_to_host_sync()?;

    let scalar_value = ScalarValue::from_proto_bytes(scalar_bytes, dtype, session)?;
    let fill_value = Scalar::try_new(dtype.clone(), scalar_value)?;

    vortex_ensure_eq!(children.len(), 2, "SparseArray expects 2 children");

    let patch_indices = children.get(
        0,
        &metadata.patches.indices_dtype()?,
        metadata.patches.len()?,
    )?;
    let patch_values = children.get(1, dtype, metadata.patches.len()?)?;

    let patches = Patches::new(
        len,
        metadata.patches.offset()?,
        patch_indices,
        patch_values,
        None,
        &mut session.create_execution_ctx(),
    )?;
    if externalize && dtype.is_int() {
        let base = ConstantArray::new(fill_value, len).into_array();
        return Ok(Patched::from_array_and_patches(
            base,
            &patches,
            &mut session.create_execution_ctx(),
        )?
        .into_array());
    }
    let slots = SparseData::make_slots(&patches);
    let data = SparseData::from_patches(&patches, fill_value)?;
    Ok(
        Array::try_from_parts(ArrayParts::new(Sparse, dtype.clone(), len, data, slots))?
            .into_array(),
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Patched;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::deserialize;
    use crate::Sparse;

    #[test]
    fn externalize_only_integer_sparse() -> VortexResult<()> {
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        for (values, fill, integer) in [
            (
                PrimitiveArray::from_option_iter([Some(-7i64), None]).into_array(),
                Scalar::from(Some(42i64)),
                true,
            ),
            (
                PrimitiveArray::from_iter([7u32, 9]).into_array(),
                Scalar::from(42u32),
                true,
            ),
            (
                PrimitiveArray::from_iter([7.25f64, 9.5]).into_array(),
                Scalar::from(42.5f64),
                false,
            ),
        ] {
            let array = Sparse::try_new(
                PrimitiveArray::from_iter([1u32, 3]).into_array(),
                values,
                5,
                fill,
                &mut ctx,
            )?
            .into_array();
            let serialized = Sparse.serialize(&array, &session)?.unwrap();
            let buffers = array
                .buffers()
                .into_iter()
                .map(BufferHandle::new_host)
                .collect::<Vec<_>>();
            let children = array.children();
            for externalize in [false, true] {
                let decoded = deserialize(
                    ArrayDeserialization::new(
                        ArrayPlugin::id(&Sparse),
                        array.dtype(),
                        array.len(),
                        &serialized.metadata,
                        &buffers,
                        &children,
                    ),
                    &session,
                    externalize,
                )?;
                assert_eq!(decoded.is::<Patched>(), externalize && integer);
                assert_eq!(decoded.is::<Sparse>(), !(externalize && integer));
                assert_arrays_eq!(decoded, array, &mut ctx);
            }
        }
        Ok(())
    }
}
