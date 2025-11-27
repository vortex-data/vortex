// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_vector::Vector;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;

use crate::ArrayOperator;
use crate::EmptyMetadata;
use crate::ToCanonical;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

mod array;
mod canonical;
mod compute;
mod operations;
mod validity;
mod visitor;

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.chunked")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ChunkedVTable.as_vtable()
    }

    fn metadata(_array: &ChunkedArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ChunkedArray> {
        if children.is_empty() {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = children.len() - 1;

        // The first child contains the row offsets of the chunks
        let chunk_offsets_array = children
            .get(
                0,
                &DType::Primitive(PType::U64, Nullability::NonNullable),
                // 1 extra offset for the end of the last chunk
                nchunks + 1,
            )?
            .to_primitive();

        let chunk_offsets_buf = chunk_offsets_array.buffer::<u64>();

        // The remaining children contain the actual data of the chunks
        let chunks = chunk_offsets_buf
            .iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (start, end))| {
                let chunk_len = usize::try_from(end - start)
                    .map_err(|_| vortex_err!("chunk_len {} exceeds usize range", end - start))?;
                children.get(idx + 1, dtype, chunk_len)
            })
            .try_collect()?;

        let chunk_offsets = PrimitiveArray::new(chunk_offsets_buf.clone(), Validity::NonNullable);

        let total_len = chunk_offsets_buf
            .last()
            .ok_or_else(|| vortex_err!("chunk_offsets must not be empty"))?;
        let len = usize::try_from(*total_len)
            .map_err(|_| vortex_err!("total length {} exceeds usize range", total_len))?;

        // Construct directly using the struct fields to avoid recomputing chunk_offsets
        Ok(ChunkedArray {
            dtype: dtype.clone(),
            len,
            chunk_offsets,
            chunks,
            stats_set: Default::default(),
        })
    }

    fn execute(array: &Self::Array, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let mut vector = VectorMut::with_capacity(array.dtype(), 0);
        for chunk in array.chunks() {
            let chunk_vector = chunk.execute_batch(ctx)?;
            vector.extend_from_vector(&chunk_vector);
        }
        Ok(vector.freeze())
    }
}

#[derive(Debug)]
pub struct ChunkedVTable;
