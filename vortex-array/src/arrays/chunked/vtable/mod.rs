// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::vtable::canonical::_canonicalize;
use crate::arrays::chunked::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

mod array;
mod canonical;
mod compute;
mod operations;
mod rules;
mod validity;
mod visitor;

vtable!(Chunked);

#[derive(Debug)]
pub struct ChunkedVTable;

impl ChunkedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.chunked");
}

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
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

        let chunk_offsets_buf = chunk_offsets_array.to_buffer::<u64>();

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

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // Children: chunk_offsets, then chunks...
        vortex_ensure!(
            !children.is_empty(),
            "Chunked array needs at least one child"
        );

        let nchunks = children.len() - 1;
        let chunk_offsets_array = children[0].to_primitive();
        let chunk_offsets_buf = chunk_offsets_array.to_buffer::<u64>();

        vortex_ensure!(
            chunk_offsets_buf.len() == nchunks + 1,
            "Expected {} chunk offsets, found {}",
            nchunks + 1,
            chunk_offsets_buf.len()
        );

        let chunks = children.into_iter().skip(1).collect();
        array.chunk_offsets = PrimitiveArray::new(chunk_offsets_buf.clone(), Validity::NonNullable);
        array.chunks = chunks;

        let total_len = chunk_offsets_buf
            .last()
            .ok_or_else(|| vortex_err!("chunk_offsets must not be empty"))?;
        array.len = usize::try_from(*total_len)
            .map_err(|_| vortex_err!("total length {} exceeds usize range", total_len))?;

        Ok(())
    }

    fn append_to_builder(
        array: &ChunkedArray,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for chunk in array.chunks() {
            chunk.append_to_builder(builder, ctx)?;
        }
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        _canonicalize(array, ctx)
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        Ok(match array.chunks.len() {
            0 => Some(Canonical::empty(array.dtype()).into_array()),
            1 => Some(array.chunks[0].clone()),
            _ => None,
        })
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        assert!(
            !array.is_empty() || (range.start > 0 && range.end > 0),
            "Empty chunked array can't be sliced from {} to {}",
            range.start,
            range.end
        );

        if array.is_empty() {
            // SAFETY: empty chunked array trivially satisfies all validations
            unsafe {
                return Ok(Some(
                    ChunkedArray::new_unchecked(vec![], array.dtype().clone()).into_array(),
                ));
            }
        }

        let (offset_chunk, offset_in_first_chunk) = array.find_chunk_idx(range.start)?;
        let (length_chunk, length_in_last_chunk) = array.find_chunk_idx(range.end)?;

        if length_chunk == offset_chunk {
            let chunk = array.chunk(offset_chunk);
            return Ok(Some(
                chunk.slice(offset_in_first_chunk..length_in_last_chunk)?,
            ));
        }

        let mut chunks = (offset_chunk..length_chunk + 1)
            .map(|i| array.chunk(i).clone())
            .collect_vec();
        if let Some(c) = chunks.first_mut() {
            *c = c.slice(offset_in_first_chunk..c.len())?;
        }

        if length_in_last_chunk == 0 {
            chunks.pop();
        } else if let Some(c) = chunks.last_mut() {
            *c = c.slice(0..length_in_last_chunk)?;
        }

        // SAFETY: chunks are slices of the original valid chunks, preserving their dtype.
        // All chunks maintain the same dtype as the original array.
        Ok(Some(unsafe {
            ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array()
        }))
    }
}
