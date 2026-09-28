// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
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
        let view = array
            .as_opt::<Sparse>()
            .ok_or_else(|| vortex_err!("Sparse plugin cannot serialize {}", array.encoding_id()))?;
        let patches = view.patches().to_metadata(view.len(), view.dtype())?;
        let metadata = SparseMetadata { patches };

        // Note that we DO NOT serialize the fill value since that is stored in the buffers.
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata,
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Sparse plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = SparseMetadata::decode(metadata)?;

        // Once we have the patches metadata, we need to get the fill value from the buffers.

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let scalar_bytes: &[u8] = &buffers[0].clone().try_to_host_sync()?;

        let scalar_value = ScalarValue::from_proto_bytes(scalar_bytes, dtype, session)?;
        let fill_value = Scalar::try_new(dtype.clone(), scalar_value)?;

        vortex_ensure_eq!(
            children.len(),
            2,
            "SparseArray expects 2 children for sparse encoding, found {}",
            children.len()
        );

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
        )?;
        let slots = SparseData::make_slots(&patches);
        let data = SparseData::from_patches(&patches, fill_value)?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
