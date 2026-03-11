// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::Precision;
use crate::ToCanonical;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::compute::kernel::PARENT_KERNELS;
use crate::arrays::chunked::compute::rules::PARENT_RULES;
use crate::arrays::chunked::vtable::canonical::_canonicalize;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
mod canonical;
mod operations;
mod validity;
vtable!(Chunked);

#[derive(Debug)]
pub struct ChunkedVTable;

impl ChunkedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.chunked");
}

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &ChunkedArray) -> usize {
        array.len
    }

    fn dtype(array: &ChunkedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ChunkedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ChunkedArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.len.hash(state);
        array
            .chunk_offsets_array()
            .as_ref()
            .array_hash(state, precision);
        for chunk in array.chunks() {
            chunk.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ChunkedArray, other: &ChunkedArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.len == other.len
            && array
                .chunk_offsets_array()
                .as_ref()
                .array_eq(other.chunk_offsets_array().as_ref(), precision)
            && array.nchunks() == other.nchunks()
            && array
                .chunks()
                .iter()
                .zip(other.chunks().iter())
                .all(|(a, b)| a.array_eq(b, precision))
    }

    fn nbuffers(_array: &ChunkedArray) -> usize {
        0
    }

    fn buffer(_array: &ChunkedArray, idx: usize) -> BufferHandle {
        vortex_panic!("ChunkedArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ChunkedArray, idx: usize) -> Option<String> {
        vortex_panic!("ChunkedArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(array: &ChunkedArray) -> usize {
        1 + array.chunks().len()
    }

    fn child(array: &ChunkedArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.chunk_offsets_array().into_array(),
            n => array.chunks()[n - 1].clone(),
        }
    }

    fn child_name(_array: &ChunkedArray, idx: usize) -> String {
        match idx {
            0 => "chunk_offsets".to_string(),
            n => format!("chunks[{}]", n - 1),
        }
    }

    fn metadata(_array: &ChunkedArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
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
        let chunks: Vec<ArrayRef> = chunk_offsets_buf
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

        // Construct directly using slots to avoid recomputing chunk_offsets
        let mut slots = Vec::with_capacity(1 + chunks.len());
        slots.push(Some(chunk_offsets.into_array()));
        slots.extend(chunks.into_iter().map(Some));
        Ok(ChunkedArray {
            dtype: dtype.clone(),
            len,
            slots,
            stats_set: Default::default(),
        })
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(_canonicalize(array, ctx)?.into_array()))
    }

    fn slots(array: &ChunkedArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ChunkedArray, idx: usize) -> &str {
        match idx {
            0 => "chunk_offsets",
            _ => "chunk",
        }
    }

    fn with_slots(array: &mut ChunkedArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        array.slots = slots;
        Ok(())
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        Ok(match array.nchunks() {
            0 => Some(Canonical::empty(array.dtype()).into_array()),
            1 => Some(array.chunk(0).clone()),
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

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}
