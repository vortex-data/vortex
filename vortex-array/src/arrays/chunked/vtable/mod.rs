// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::chunked::ChunkedData;
use crate::arrays::chunked::array::CHUNK_OFFSETS_SLOT;
use crate::arrays::chunked::array::CHUNKS_OFFSET;
use crate::arrays::chunked::compute::kernel::PARENT_KERNELS;
use crate::arrays::chunked::compute::rules::PARENT_RULES;
use crate::arrays::chunked::vtable::canonical::_canonicalize;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
mod canonical;
mod operations;
mod validity;

/// A [`Chunked`]-encoded Vortex array.
pub type ChunkedArray = Array<Chunked>;

#[derive(Clone, Debug)]
pub struct Chunked;

impl ArrayHash for ChunkedData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {
        // Chunk offsets are cached derived data. Slot 0 already stores the logical offsets array,
        // and ArrayInner hashing includes every slot before ArrayData.
    }
}

impl ArrayEq for ChunkedData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        // Chunk offsets are cached derived data. Slot 0 already stores the logical offsets array,
        // and ArrayInner equality compares every slot before ArrayData.
        true
    }
}

impl VTable for Chunked {
    type ArrayData = ChunkedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.chunked");
        *ID
    }

    fn validate(
        &self,
        data: &ChunkedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            !slots.is_empty(),
            "ChunkedArray must have at least a chunk offsets slot"
        );
        let chunk_offsets = slots[CHUNK_OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("validated chunk offsets slot");
        vortex_ensure!(
            chunk_offsets.dtype() == &DType::Primitive(PType::U64, Nullability::NonNullable),
            "ChunkedArray chunk offsets must be non-nullable u64, found {}",
            chunk_offsets.dtype()
        );
        vortex_ensure!(
            chunk_offsets.len() == data.chunk_offsets.len(),
            "ChunkedArray chunk offsets slot length {} does not match cached offsets length {}",
            chunk_offsets.len(),
            data.chunk_offsets.len()
        );
        vortex_ensure!(
            data.chunk_offsets.len() == slots.len() - CHUNKS_OFFSET + 1,
            "ChunkedArray chunk offsets length {} does not match {} chunks",
            data.chunk_offsets.len(),
            slots.len() - CHUNKS_OFFSET
        );
        vortex_ensure!(
            data.chunk_offsets
                .last()
                .copied()
                .vortex_expect("chunked arrays always have a leading 0 offset")
                == len,
            "ChunkedArray length {} does not match outer length {}",
            data.chunk_offsets.last().copied().unwrap_or_default(),
            len
        );
        for (idx, (start, end)) in data
            .chunk_offsets
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
        {
            let chunk = slots[CHUNKS_OFFSET + idx]
                .as_ref()
                .vortex_expect("validated chunk slot");
            vortex_ensure!(
                chunk.dtype() == dtype,
                "ChunkedArray chunk dtype {} does not match outer dtype {}",
                chunk.dtype(),
                dtype
            );
            vortex_ensure!(
                chunk.len() == end - start,
                "ChunkedArray chunk {} len {} does not match offsets span {}",
                idx,
                chunk.len(),
                end - start
            );
        }
        Ok(())
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

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
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
            CHUNK_OFFSETS_SLOT,
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            nchunks + 1,
        )?;
        #[expect(deprecated)]
        let chunk_offsets_buf = chunk_offsets.to_primitive().to_buffer::<u64>();
        let chunk_offsets_usize = chunk_offsets_buf
            .iter()
            .copied()
            .map(|offset| {
                usize::try_from(offset)
                    .map_err(|_| vortex_err!("chunk offset {offset} exceeds usize range"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let mut slots = Vec::with_capacity(children.len());
        slots.push(Some(chunk_offsets));
        for (idx, (start, end)) in chunk_offsets_usize
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
        {
            let chunk_len = end - start;
            slots.push(Some(children.get(idx + CHUNKS_OFFSET, dtype, chunk_len)?));
        }

        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ChunkedData::new(chunk_offsets_usize),
        )
        .with_slots(slots))
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            CHUNK_OFFSETS_SLOT => "chunk_offsets".to_string(),
            n => format!("chunks[{}]", n - CHUNKS_OFFSET),
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        match array.dtype() {
            // Struct and List need special swizzling logic, use the existing canonicalize path.
            DType::Struct(..) | DType::List(..) => {
                // TODO(joe)[#7674]: iterative execution here too
                Ok(ExecutionResult::done(_canonicalize(array.as_view(), ctx)?))
            }
            // For all other types, use the builder path via AppendChild.
            _ => {
                let slot_idx = array.next_builder_slot.max(CHUNKS_OFFSET);
                if slot_idx < array.slots().len() {
                    Ok(ExecutionResult::append_child(
                        array.with_next_builder_slot(slot_idx + 1),
                        slot_idx,
                    ))
                } else {
                    Ok(ExecutionResult::done(
                        Canonical::empty(array.dtype()).into_array(),
                    ))
                }
            }
        }
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        Ok(match array.nchunks() {
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
}
