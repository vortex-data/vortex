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
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::Zstd;
use super::ZstdData;
use crate::ZstdMetadata;

impl ArrayPlugin for Zstd {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Zstd>()
            .ok_or_else(|| vortex_err!("Zstd plugin cannot serialize {}", array.encoding_id()))?;
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            view.metadata.clone().encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
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
            "Zstd plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = ZstdMetadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("ZstdArray expected 0 or 1 child, got {}", children.len());
        };

        let (dictionary_buffer, compressed_buffers) = if metadata.dictionary_size == 0 {
            // no dictionary
            (
                None,
                buffers
                    .iter()
                    .map(|b| b.clone().try_to_host_sync())
                    .collect::<VortexResult<Vec<_>>>()?,
            )
        } else {
            // with dictionary
            (
                Some(buffers[0].clone().try_to_host_sync()?),
                buffers[1..]
                    .iter()
                    .map(|b| b.clone().try_to_host_sync())
                    .collect::<VortexResult<Vec<_>>>()?,
            )
        };

        let slots = smallvec![validity_to_child(&validity, len)];
        let data = ZstdData::new(dictionary_buffer, compressed_buffers, metadata, len);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
