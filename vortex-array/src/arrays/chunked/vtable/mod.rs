// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ToCanonical;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedData;
use crate::arrays::chunked::array::CHUNK_OFFSETS_SLOT;
use crate::arrays::chunked::array::CHUNKS_OFFSET;
use crate::arrays::chunked::compute::kernel::PARENT_KERNELS;
use crate::arrays::chunked::compute::rules::PARENT_RULES;
use crate::arrays::chunked::vtable::canonical::_canonicalize;
use crate::arrays::primitive::PrimitiveData;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
mod canonical;
mod operations;
mod validity;
vtable!(Chunked, Chunked, ChunkedData);

#[derive(Clone, Debug)]
pub struct Chunked;

impl Chunked {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.chunked");
}

impl VTable for Chunked {
    type ArrayData = ChunkedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn array_hash<H: std::hash::Hasher>(array: &ChunkedData, state: &mut H, precision: Precision) {
        PrimitiveArray::new(
            array.chunk_offsets.to_buffer::<u64>(),
            Validity::NonNullable,
        )
        .into_array()
        .array_hash(state, precision);
        for chunk in &array.chunks {
            chunk.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ChunkedData, other: &ChunkedData, precision: Precision) -> bool {
        PrimitiveArray::new(
            array.chunk_offsets.to_buffer::<u64>(),
            Validity::NonNullable,
        )
        .into_array()
        .array_eq(
            &PrimitiveArray::new(
                other.chunk_offsets.to_buffer::<u64>(),
                Validity::NonNullable,
            )
            .into_array(),
            precision,
        ) && array.chunks.len() == other.chunks.len()
            && array
                .iter_chunks()
                .zip(other.iter_chunks())
                .all(|(a, b)| a.array_eq(b, precision))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ChunkedArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ChunkedArray buffer_name index {idx} out of bounds")
    }

    fn serialize(_array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn validate(&self, data: &ChunkedData, dtype: &DType, len: usize) -> VortexResult<()> {
        vortex_ensure!(
            data.len() == len,
            "ChunkedArray length {} does not match outer length {}",
            data.len(),
            len
        );
        vortex_ensure!(
            data.dtype() == dtype,
            "ChunkedArray dtype {} does not match outer dtype {}",
            data.dtype(),
            dtype
        );
        Ok(())
    }

    fn deserialize(
        &self,
        dtype: &DType,
        _len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ChunkedData> {
        if !metadata.is_empty() {
            vortex_bail!("ChunkedArray expects empty metadata, got {} bytes", metadata.len());
        }
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

        let chunk_offsets = PrimitiveData::new(chunk_offsets_buf.clone(), Validity::NonNullable);

        let slots = ChunkedData::make_slots(&chunk_offsets, &chunks);
        // Construct directly using the struct fields to avoid recomputing chunk_offsets
        Ok(ChunkedData {
            empty_dtype: chunks.is_empty().then_some(dtype.clone()),
            chunk_offsets,
            chunks,
            slots,
        })
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            CHUNK_OFFSETS_SLOT => "chunk_offsets".to_string(),
            n => format!("chunks[{}]", n - CHUNKS_OFFSET),
        }
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        // Slots: chunk_offsets, then chunks...
        vortex_ensure!(!slots.is_empty(), "Chunked array needs at least one slot");

        let nchunks = slots.len() - CHUNKS_OFFSET;
        let chunk_offsets_ref = slots[CHUNK_OFFSETS_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("chunk_offsets slot must not be None"))?;
        let chunk_offsets_buf = chunk_offsets_ref.to_primitive().to_buffer::<u64>();

        vortex_ensure!(
            chunk_offsets_buf.len() == nchunks + 1,
            "Expected {} chunk offsets, found {}",
            nchunks + 1,
            chunk_offsets_buf.len()
        );

        let chunks: Vec<ArrayRef> = slots[CHUNKS_OFFSET..]
            .iter()
            .map(|s| {
                s.clone()
                    .ok_or_else(|| vortex_err!("chunk slot must not be None"))
            })
            .try_collect()?;
        array.chunk_offsets = PrimitiveData::new(chunk_offsets_buf.clone(), Validity::NonNullable);
        array.empty_dtype = chunks.is_empty().then_some(array.dtype().clone());
        array.chunks = chunks;
        array.slots = slots;

        Ok(())
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for chunk in array.iter_chunks() {
            chunk.append_to_builder(builder, ctx)?;
        }
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(_canonicalize(array.as_view(), ctx)?))
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        Ok(match array.chunks.len() {
            0 => Some(Canonical::empty(array.dtype()).into_array()),
            1 => Some(array.chunk(0).clone()),
            _ => None,
        })
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}
