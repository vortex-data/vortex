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
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Sparse;
use crate::SparseData;
use crate::SparseExt;
use crate::SparseMetadata;

/// The frozen sparse wire ID: patch indices and values, without chunk offsets.
pub fn sparse_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.sparse");
    *ID
}

/// The experimental sparse wire ID for arrays whose patches carry chunk offsets.
///
/// Its children are the patch indices, the patch values and then the chunk offsets, when the
/// metadata declares them. Readers ignore any further children, so later revisions can add
/// children without breaking existing readers.
pub fn sparse_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.sparse.v2");
    *ID
}

/// Serde for [`Sparse`] using the frozen v1 and experimental v2 wire formats.
///
/// The plugin writes v1 whenever the patches have no chunk offsets, so such arrays stay readable
/// by older readers, and v2 otherwise.
impl ArrayPlugin for Sparse {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![sparse_v1_id(), sparse_v2_id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![sparse_v1_id(), sparse_v2_id()]
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
        let patches = array.patches();
        let serialized_id = if patches.chunk_offsets().is_some() {
            sparse_v2_id()
        } else {
            sparse_v1_id()
        };
        let metadata = SparseMetadata {
            patches: patches.to_metadata(array.len(), array.dtype())?,
        };

        // Note that we DO NOT serialize the fill value since that is stored in the buffers.
        Ok(Some(ArraySerialization::from_array(
            serialized_id,
            array.array(),
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let serialized_id = parts.serialized_id;
        vortex_ensure!(
            serialized_id == sparse_v1_id() || serialized_id == sparse_v2_id(),
            "Sparse plugin does not recognize serialized ID {serialized_id}",
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

        let chunk_offsets_dtype = metadata.patches.chunk_offsets_dtype()?;
        if serialized_id == sparse_v1_id() {
            vortex_ensure_eq!(children.len(), 2, "SparseArray expects 2 children");
            vortex_ensure!(
                chunk_offsets_dtype.is_none(),
                "{serialized_id} does not support patch chunk offsets"
            );
        }

        let patch_indices = children.get(
            0,
            &metadata.patches.indices_dtype()?,
            metadata.patches.len()?,
        )?;
        let patch_values = children.get(1, dtype, metadata.patches.len()?)?;

        let patch_chunk_offsets = chunk_offsets_dtype
            .map(|dtype| {
                children.get(
                    2,
                    &dtype,
                    usize::try_from(metadata.patches.chunk_offsets_len())?,
                )
            })
            .transpose()?;

        let patches = Patches::from_metadata(
            len,
            &metadata.patches,
            patch_indices,
            patch_values,
            patch_chunk_offsets,
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
