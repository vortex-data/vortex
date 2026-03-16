// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use fastlanes::FastLanes;
use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ChildRangeRead;
use vortex_array::vtable::EncodingRangeRead;
use vortex_array::vtable::RangeDecodeInfo;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::DeltaArray;
use crate::delta::array::delta_decompress::delta_decompress;

mod operations;
mod rules;
mod slice;
mod validity;

vtable!(Delta);

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl VTable for Delta {
    type Array = DeltaArray;

    type Metadata = ProstMetadata<DeltaMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &DeltaArray) -> usize {
        array.len()
    }

    fn dtype(array: &DeltaArray) -> &DType {
        array.dtype()
    }

    fn stats(array: &DeltaArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DeltaArray, state: &mut H, precision: Precision) {
        array.offset().hash(state);
        array.len().hash(state);
        array.dtype().hash(state);
        array.bases().array_hash(state, precision);
        array.deltas().array_hash(state, precision);
    }

    fn array_eq(array: &DeltaArray, other: &DeltaArray, precision: Precision) -> bool {
        array.offset() == other.offset()
            && array.len() == other.len()
            && array.dtype() == other.dtype()
            && array.bases().array_eq(other.bases(), precision)
            && array.deltas().array_eq(other.deltas(), precision)
    }

    fn nbuffers(_array: &DeltaArray) -> usize {
        0
    }

    fn buffer(_array: &DeltaArray, idx: usize) -> BufferHandle {
        vortex_panic!("DeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &DeltaArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &DeltaArray) -> usize {
        2
    }

    fn child(array: &DeltaArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.bases().clone(),
            1 => array.deltas().clone(),
            _ => vortex_panic!("DeltaArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &DeltaArray, idx: usize) -> String {
        match idx {
            0 => "bases".to_string(),
            1 => "deltas".to_string(),
            _ => vortex_panic!("DeltaArray child name index {idx} out of bounds"),
        }
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // DeltaArray children order (from visit_children):
        // 1. bases
        // 2. deltas

        vortex_ensure!(
            children.len() == 2,
            "Expected 2 children for Delta encoding, got {}",
            children.len()
        );

        array.bases = children[0].clone();
        array.deltas = children[1].clone();

        Ok(())
    }

    fn metadata(array: &DeltaArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata::decode(bytes)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DeltaArray> {
        assert_eq!(children.len(), 2);
        let ptype = PType::try_from(dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |T| { <T as FastLanes>::LANES });

        // Compute the length of the deltas array from len + offset rather than metadata.
        // This allows range reads to work with sub-ranged children, where the buffer is
        // shorter than the original metadata.deltas_len.
        let deltas_len = len + metadata.0.offset as usize;
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if !deltas_len.is_multiple_of(1024) {
            1
        } else {
            0
        };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        DeltaArray::try_new(bases, deltas, metadata.0.offset as usize, len)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(
            delta_decompress(array, ctx)?.into_array(),
        ))
    }

    fn plan_range_read(
        metadata: &ProstMetadata<DeltaMetadata>,
        row_range: Range<usize>,
        _row_count: usize,
        dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        if metadata.0.offset != 0 {
            return None;
        }

        let deltas_len = usize::try_from(metadata.0.deltas_len).ok()?;
        let byte_width = match dtype {
            DType::Primitive(ptype, _) => ptype.byte_width(),
            _ => return None,
        };
        let lanes = match byte_width {
            1 => 128,
            2 => 64,
            4 => 32,
            8 => 16,
            _ => return None,
        };

        let first_chunk = row_range.start / 1024;
        let last_chunk = row_range.end.saturating_sub(1) / 1024;

        // Child 1 = deltas (row-indexed, same dtype).
        let deltas_row_start = first_chunk * 1024;
        let deltas_row_end = ((last_chunk + 1) * 1024).min(deltas_len);

        // Child 0 = bases (LANES values per full chunk, 1 per remainder).
        let num_full_chunks = deltas_len / 1024;
        let has_remainder = !deltas_len.is_multiple_of(1024);
        let bases_len = num_full_chunks * lanes + if has_remainder { 1 } else { 0 };
        let bases_row_start = first_chunk * lanes;
        let bases_row_end = if last_chunk >= num_full_chunks {
            bases_len
        } else {
            (last_chunk + 1) * lanes
        }
        .min(bases_len);

        let sub_deltas_len = deltas_row_end - deltas_row_start;
        let intra_chunk_offset = row_range.start - deltas_row_start;
        let post_slice = (intra_chunk_offset > 0 || sub_deltas_len > row_range.len())
            .then(|| intra_chunk_offset..(intra_chunk_offset + row_range.len()));

        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![],
            children: vec![
                // Child 0 = bases.
                ChildRangeRead::Recurse {
                    row_range: bases_row_start..bases_row_end,
                    row_count: bases_len,
                    dtype: dtype.clone(),
                },
                // Child 1 = deltas.
                ChildRangeRead::Recurse {
                    row_range: deltas_row_start..deltas_row_end,
                    row_count: deltas_len,
                    dtype: dtype.clone(),
                },
            ],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len: sub_deltas_len,
                post_slice,
            },
        })
    }
}

#[derive(Debug)]
pub struct Delta;

impl Delta {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.delta");
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;
    use super::ProstMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            ProstMetadata(DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }),
        );
    }
}
