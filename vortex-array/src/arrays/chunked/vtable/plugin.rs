// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
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
        vortex_ensure!(
            array.is::<Chunked>(),
            "Chunked plugin cannot serialize {}",
            array.encoding_id()
        );
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
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
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Chunked plugin does not recognize serialized ID {serialized_id}"
        );
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

        Ok(Array::try_from_parts(
            ArrayParts::new(
                self.clone(),
                dtype.clone(),
                len,
                ChunkedData::new(chunk_offsets_usize),
            )
            .with_slots(slots),
        )?
        .into_array())
    }
}
