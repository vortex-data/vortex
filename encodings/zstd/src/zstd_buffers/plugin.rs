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
use vortex_array::ArraySlots;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ZstdBuffers;
use super::ZstdBuffersData;
use super::array_id_from_string;
use crate::ZstdBuffersMetadata;

impl ArrayPlugin for ZstdBuffers {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<ZstdBuffers>().ok_or_else(|| {
            vortex_err!(
                "ZstdBuffers plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let children: Vec<&ArrayRef> = view.slots().iter().flatten().collect();
        let child_dtypes = children
            .iter()
            .map(|child| child.dtype().try_into())
            .collect::<VortexResult<Vec<_>>>()?;
        let child_lens = children.iter().map(|child| child.len() as u64).collect();

        let metadata = ZstdBuffersMetadata {
            inner_encoding_id: view.inner_encoding_id.to_string(),
            inner_metadata: view.inner_metadata.clone(),
            uncompressed_sizes: view.uncompressed_sizes.clone(),
            buffer_alignments: view.buffer_alignments.clone(),
            child_dtypes,
            child_lens,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
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
            "ZstdBuffers plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = ZstdBuffersMetadata::decode(metadata)?;
        let compressed_buffers: Vec<BufferHandle> = buffers.to_vec();

        // Children belong to inner encodings, and serialization doesn't
        // preserve their dtypes and values. Check dtypes are recovered from
        // metadata.
        vortex_ensure_eq!(metadata.child_dtypes.len(), children.len());
        vortex_ensure_eq!(metadata.child_lens.len(), children.len());

        let slots: ArraySlots = (0..children.len())
            .map(|i| {
                let child_dtype = DType::from_proto(&metadata.child_dtypes[i], session)?;
                let child_len = usize::try_from(metadata.child_lens[i])?;
                children.get(i, &child_dtype, child_len).map(Some)
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into();

        let data = ZstdBuffersData {
            inner_encoding_id: array_id_from_string(&metadata.inner_encoding_id),
            inner_metadata: metadata.inner_metadata.clone(),
            compressed_buffers,
            uncompressed_sizes: metadata.uncompressed_sizes.clone(),
            buffer_alignments: metadata.buffer_alignments.clone(),
        };

        data.validate()?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
