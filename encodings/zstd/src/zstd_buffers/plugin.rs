// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`ZstdBuffers`].

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
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::ZstdBuffersData;
use super::array_id_from_string;
use super::lift_validity;
use crate::ZstdBuffers;
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
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "ZstdBuffers plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<ZstdBuffers>();
        let children: Vec<&ArrayRef> = array.slots().iter().flatten().collect();
        let child_dtypes = children
            .iter()
            .map(|child| child.dtype().try_into())
            .collect::<VortexResult<Vec<_>>>()?;
        let child_lens = children.iter().map(|child| child.len() as u64).collect();

        let metadata = ZstdBuffersMetadata {
            inner_encoding_id: array.inner_encoding_id.to_string(),
            inner_metadata: array.inner_metadata.clone(),
            uncompressed_sizes: array.uncompressed_sizes.clone(),
            buffer_alignments: array.buffer_alignments.clone(),
            child_dtypes,
            child_lens,
            inner_child_count: Some(u32::try_from(array.inner_child_count)?),
            validity_child: array.validity_slot.map(u32::try_from).transpose()?,
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
            "ZstdBuffers plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        let metadata = ZstdBuffersMetadata::decode(metadata)?;
        let compressed_buffers: Vec<BufferHandle> = buffers.to_vec();

        // Children belong to inner encodings, and serialization doesn't
        // preserve their dtypes and values. Check dtypes are recovered from
        // metadata.
        vortex_ensure_eq!(metadata.child_dtypes.len(), children.len());
        vortex_ensure_eq!(metadata.child_lens.len(), children.len());

        let mut slots = (0..children.len())
            .map(|i| {
                let child_dtype = DType::from_proto(&metadata.child_dtypes[i], session)?;
                let child_len = usize::try_from(metadata.child_lens[i])?;
                children.get(i, &child_dtype, child_len).map(Some)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let mut data = ZstdBuffersData {
            inner_encoding_id: array_id_from_string(&metadata.inner_encoding_id),
            inner_metadata: metadata.inner_metadata.clone(),
            compressed_buffers,
            uncompressed_sizes: metadata.uncompressed_sizes.clone(),
            buffer_alignments: metadata.buffer_alignments.clone(),
            inner_child_count: children.len(),
            validity_slot: None,
        };
        data.validate()?;

        match metadata.inner_child_count {
            Some(inner_child_count) => {
                data.inner_child_count = usize::try_from(inner_child_count)?;
                data.validity_slot = metadata.validity_child.map(usize::try_from).transpose()?;
            }
            // Arrays written before the validity was lifted out of the wrapped array only hold
            // its children, so rebuild it once here to find its validity.
            None if dtype.is_nullable() => {
                let legacy = Array::try_from_parts(ArrayParts::new(
                    self.clone(),
                    dtype.clone(),
                    len,
                    data.clone(),
                    slots.clone().into(),
                ))?;
                let validity =
                    ZstdBuffers::decompress_and_build_inner(&legacy, session)?.validity()?;
                data.validity_slot = lift_validity(&validity, len, &mut slots);
            }
            None => {}
        }

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots.into(),
        ))?
        .into_array())
    }
}
