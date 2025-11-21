// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_vector::{Vector, VectorMut, VectorMutOps};

use crate::arrays::{ChunkedArray, PrimitiveArray};
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{NotSupported, VTable};
use crate::{ArrayOperator, EmptyMetadata, EncodingId, EncodingRef, ToCanonical, vtable};

mod array;
mod canonical;
mod compute;
mod operations;
mod validity;
mod visitor;

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;
    type Encoding = ChunkedEncoding;
    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.chunked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ChunkedEncoding.as_ref())
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
        _encoding: &ChunkedEncoding,
        dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
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

#[derive(Clone, Debug)]
pub struct ChunkedEncoding;
