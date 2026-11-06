// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Chunked`].

use itertools::Itertools;
use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::arrays::Chunked;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedData;
use crate::arrays::chunked::array::ChunkedSlots;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

impl ArrayPlugin for Chunked {
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
            "Chunked plugin cannot serialize an array of another encoding",
        );
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
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
            "Chunked plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        if !metadata.is_empty() {
            vortex_bail!(
                "ChunkedArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if children.is_empty() {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = children.len() - 1;
        let chunk_offsets = children.get(
            ChunkedSlots::CHUNK_OFFSETS,
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            nchunks + 1,
        )?;
        let mut ctx = session.create_execution_ctx();
        let chunk_offsets_buf = chunk_offsets
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .to_buffer::<u64>();
        let chunk_offsets_usize = chunk_offsets_buf
            .iter()
            .copied()
            .map(|offset| {
                usize::try_from(offset)
                    .map_err(|_| vortex_err!("chunk offset {offset} exceeds usize range"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let mut slots = SmallVec::with_capacity(children.len());
        slots.push(Some(chunk_offsets));
        for (idx, (start, end)) in chunk_offsets_usize
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
        {
            let chunk_len = end - start;
            slots.push(Some(children.get(
                idx + ChunkedSlots::CHUNKS_OFFSET,
                dtype,
                chunk_len,
            )?));
        }

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ChunkedData::new(chunk_offsets_usize),
            slots,
        ))?
        .into_array())
    }
}
