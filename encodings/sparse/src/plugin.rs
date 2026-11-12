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
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
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
        let slots = SparseData::make_slots(&patches);
        let data = SparseData::from_patches(&patches, fill_value)?;
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
