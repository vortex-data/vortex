// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ToCanonical;
use crate::arrays::ChunkedData;
use crate::arrays::PrimitiveData;
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
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::ArrayView;
use crate::vtable::VTable;
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

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Chunked
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ChunkedData) -> usize {
        array.len
    }

    fn dtype(array: &ChunkedData) -> &DType {
        &array.dtype
    }

    fn stats(array: &ChunkedData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: ArrayView<'_, Self>,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.len.hash(state);
        array
            .chunk_offsets
            .clone()
            .into_array()
            .array_hash(state, precision);
        for chunk in &array.chunks {
            chunk.array_hash(state, precision);
        }
    }

    fn array_eq(
        array: ArrayView<'_, Self>,
        other: ArrayView<'_, Self>,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.len == other.len
            && array
                .chunk_offsets
                .clone()
                .into_array()
                .array_eq(&other.chunk_offsets.clone().into_array(), precision)
            && array.chunks.len() == other.chunks.len()
            && array
                .chunks
                .iter()
                .zip(&other.chunks)
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

    fn nchildren(array: ArrayView<'_, Self>) -> usize {
        1 + array.chunks().len()
    }

    fn child(array: ArrayView<'_, Self>, idx: usize) -> ArrayRef {
        match idx {
            0 => array.chunk_offsets.clone().into_array(),
            n => array.chunks()[n - 1].clone(),
        }
    }

    fn child_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "chunk_offsets".to_string(),
            n => format!("chunks[{}]", n - 1),
        }
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<ChunkedData> {
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

        let chunk_offsets = PrimitiveData::new(chunk_offsets_buf.clone(), Validity::NonNullable);

        let total_len = chunk_offsets_buf
            .last()
            .ok_or_else(|| vortex_err!("chunk_offsets must not be empty"))?;
        let len = usize::try_from(*total_len)
            .map_err(|_| vortex_err!("total length {} exceeds usize range", total_len))?;

        // Construct directly using the struct fields to avoid recomputing chunk_offsets
        Ok(ChunkedData {
            dtype: dtype.clone(),
            len,
            chunk_offsets,
            chunks,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()> {
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
        array.chunk_offsets = PrimitiveData::new(chunk_offsets_buf.clone(), Validity::NonNullable);
        array.chunks = chunks;

        let total_len = chunk_offsets_buf
            .last()
            .ok_or_else(|| vortex_err!("chunk_offsets must not be empty"))?;
        array.len = usize::try_from(*total_len)
            .map_err(|_| vortex_err!("total length {} exceeds usize range", total_len))?;

        Ok(())
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for chunk in array.chunks() {
            chunk.append_to_builder(builder, ctx)?;
        }
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            _canonicalize(&array, ctx)?.into_array(),
        ))
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        Ok(match array.chunks.len() {
            0 => Some(Canonical::empty(array.dtype()).into_array()),
            1 => Some(array.chunks[0].clone()),
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
