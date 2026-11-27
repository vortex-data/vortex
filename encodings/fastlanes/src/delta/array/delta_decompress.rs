// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use fastlanes::Delta as FastLanesDelta;
use fastlanes::FastLanes;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::DeltaArray;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::chunks::DeltaChunks;

pub fn delta_decompress(
    array: &DeltaArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + array.len();

    let validity = array.validity()?;

    let original_ptype = bases.ptype();
    // Signed inputs are processed through their unsigned counterpart; `wrapping_add` on the
    // raw bytes inverts the `wrapping_sub` done at compress time regardless of signedness.
    let bases = bases.reinterpret_cast(original_ptype.to_unsigned());

    let decoded = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
        let buffer =
            DeltaChunks::<T, { T::LANES }>::for_array(array.as_view(), bases.to_buffer(), ctx)?
                .decode_all();
        // Whole chunks were decoded; keep the values within the Delta array's slice.
        let buffer = buffer.slice(start..end);

        PrimitiveArray::new(buffer, validity)
    });

    Ok(decoded.reinterpret_cast(original_ptype))
}

/// Stream decoded chunks into a parent without allocating Delta's full logical output.
pub(crate) fn delta_source(
    array: ArrayView<'_, crate::Delta>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    if !array.deltas().is::<Primitive>() && PackedDeltas::try_new(array.deltas()).is_none() {
        return Ok(None);
    }
    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let bases = bases.reinterpret_cast(bases.ptype().to_unsigned());
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    Ok(Some(match_each_unsigned_integer_ptype!(
        bases.ptype(),
        |T| {
            Box::new(DeltaSource {
                chunks: DeltaChunks::<T, { T::LANES }>::for_array(array, bases.to_buffer(), ctx)?,
                offset: array.offset(),
                validity,
            }) as Box<dyn BlockSource>
        }
    )))
}

struct DeltaSource<T, const LANES: usize> {
    chunks: DeltaChunks<T, LANES>,
    offset: usize,
    validity: Mask,
}

impl<T: NativePType + FastLanesDelta, const LANES: usize> BlockSource for DeltaSource<T, LANES> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn block_offset(&self) -> usize {
        self.offset
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let start = self.offset + rows.start;
        let chunk = scratch
            .values_mut::<T>(FL_CHUNK_SIZE)
            .first_chunk_mut()
            .vortex_expect("the scratch fits a Delta chunk");
        self.chunks.decode_into(start / FL_CHUNK_SIZE, chunk);
        Ok(SourceBlock::Scratch {
            offset: start % FL_CHUNK_SIZE,
            len: rows.len(),
        })
    }
}

/// Bit-packed deltas whose delta chunks each line up with one packed block.
///
/// Delta slices its deltas at chunk boundaries, so a BitPacked child, or a lazy slice of one, starts
/// on a packed block. The fused decode then reads whole packed blocks, and the Delta array's own
/// offset is applied by slicing the decoded chunks at the end.
#[derive(Clone, Copy)]
pub(crate) struct PackedDeltas<'a> {
    packed: ArrayView<'a, BitPacked>,
    /// The packed block holding the first delta.
    first_block: usize,
}

impl<'a> PackedDeltas<'a> {
    /// Returns the packed blocks behind `deltas` when it is a BitPacked array, or a slice of one,
    /// that starts on a packed block.
    pub(crate) fn try_new(deltas: &'a ArrayRef) -> Option<Self> {
        let (packed, slice_start) = match deltas.as_opt::<BitPacked>() {
            Some(packed) => (packed, 0),
            None => {
                let slice = deltas.as_opt::<Slice>()?;
                let start = slice.data().slice_range().start;
                let child = slice.child();
                (child.as_opt::<BitPacked>()?, start)
            }
        };
        let start = usize::from(packed.offset()) + slice_start;
        start.is_multiple_of(FL_CHUNK_SIZE).then_some(Self {
            packed,
            first_block: start / FL_CHUNK_SIZE,
        })
    }

    /// The bit-packed array holding the deltas.
    pub(crate) fn packed(&self) -> ArrayView<'a, BitPacked> {
        self.packed
    }

    /// The packed block holding the first delta.
    pub(crate) fn first_block(&self) -> usize {
        self.first_block
    }
}
