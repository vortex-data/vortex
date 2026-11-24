// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::Dict;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::Slice;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::slice::SliceArraySlotsExt;
use crate::canonical::AnyCanonical;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::validity::Validity;

/// A take of at most `1 / SPARSE_TAKE_RATIO` of the rows of a slice reads the child directly.
///
/// A take of more rows keeps the slice. Then a kernel that decodes all of its input decodes only
/// the rows of the slice, not all rows of the child.
const SPARSE_TAKE_RATIO: usize = 8;

impl TakeExecute for Slice {
    /// Take rows of the child at the indices moved by the start of the slice.
    ///
    /// The slice of an encoded child, such as an array with patches, can cost more than a take of
    /// a few rows. A file scan gives slices of larger arrays, so a sparse take, such as a random
    /// batch of rows, does not slice the child.
    fn take(
        array: ArrayView<'_, Slice>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.len().saturating_mul(SPARSE_TAKE_RATIO) > array.as_ref().len() {
            return Ok(None);
        }

        // Runs of indices keep their structure only through the slice.
        if indices.is::<PiecewiseSequence>() {
            return Ok(None);
        }

        // A child with no take kernel decodes in full for a take, so it keeps the slice, which
        // decodes only the rows of the slice.
        let child = array.child();

        if !supports_take(child, array.slice_range().clone(), ctx)? {
            return Ok(None);
        }

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = indices.validity()?;

        // An index past the end of the slice can be in the child, so the indices are checked here.
        // A null index can have any value, so indices with nulls keep the slice.
        if !matches!(validity, Validity::NonNullable | Validity::AllValid) {
            return Ok(None);
        }

        let len = array.as_ref().len() as u64;
        let start = array.slice_range().start as u64;

        let moved = match_each_integer_ptype!(indices.ptype(), |T| {
            shift_indices(indices.as_slice::<T>(), start, len)
        })?;

        let moved = PrimitiveArray::new(moved, validity).into_array();

        child.take(moved).map(Some)
    }
}

fn supports_take(array: &ArrayRef, range: Range<usize>, ctx: &ExecutionCtx) -> VortexResult<bool> {
    if let Some(chunked) = array.as_opt::<Chunked>() {
        if chunked.is_empty() || range.is_empty() {
            return Ok(true);
        }

        let (first_chunk, first_offset) = chunked.find_chunk_idx(range.start)?;
        let (last_chunk, last_offset) = chunked.find_chunk_idx(range.end - 1)?;

        for idx in first_chunk..=last_chunk {
            let chunk = chunked.chunk(idx);
            let start = if idx == first_chunk { first_offset } else { 0 };
            let end = if idx == last_chunk {
                last_offset + 1
            } else {
                chunk.len()
            };
            if !supports_take(chunk, start..end, ctx)? {
                return Ok(false);
            }
        }

        return Ok(true);
    }

    Ok(array.is::<AnyCanonical>()
        || ctx
            .session()
            .kernels()
            .has_execute_parent(Dict.id(), array.encoding_id()))
}

fn shift_indices<T: AsPrimitive<u64>>(
    indices: &[T],
    start: u64,
    len: u64,
) -> VortexResult<Buffer<u64>> {
    indices
        .iter()
        .map(|&index| {
            let index: u64 = index.as_();

            if index >= len {
                vortex_bail!("take index {index} out of bounds for length {len}");
            }

            Ok(index + start)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Slice;
    use crate::arrays::SliceArray;
    use crate::arrays::dict::TakeExecute;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// A chunked array of 64 rows. A slice of it stays a slice until execution, as the slice of
    /// an encoded array of a file does.
    fn child() -> ArrayRef {
        let rows = |range: std::ops::Range<i32>| {
            PrimitiveArray::new(
                range.clone().collect::<Buffer<i32>>(),
                Validity::from_iter(range.map(|row| row % 7 != 3)),
            )
            .into_array()
        };

        let chunks = vec![rows(0..24), rows(24..64)];
        let dtype = chunks[0].dtype().clone();

        ChunkedArray::try_new(chunks, dtype)
            .vortex_expect("chunks have one dtype")
            .into_array()
    }

    #[test]
    fn test_take_slice_reads_child() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let child = child();

        for range in [0..64, 10..42, 50..64] {
            let sliced = SliceArray::try_new(child.clone(), range.clone())?.into_array();
            let flat = child
                .slice(range)?
                .execute::<PrimitiveArray>(&mut ctx)?
                .into_array();

            for indices in [
                buffer![1u8].into_array(),
                buffer![3u16, 0, 3].into_array(),
                PrimitiveArray::from_option_iter([Some(2i64), None, Some(0)]).into_array(),
            ] {
                let result = sliced.take(indices.clone())?;
                assert_arrays_eq!(result, flat.take(indices)?, &mut ctx);
            }
        }

        Ok(())
    }

    /// An index past the end of the slice is an error, also when it is in the child.
    #[test]
    fn test_take_slice_out_of_bounds() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let sliced = SliceArray::try_new(child(), 10..42)?.into_array();

        let result = sliced.take(buffer![0u32, 32].into_array())?;
        assert!(result.execute::<Canonical>(&mut ctx).is_err());

        Ok(())
    }

    /// A slice over a chunked child pushes down a sparse take only when every overlapping chunk
    /// supports take; chunks without a take kernel keep the slice to avoid decoding the whole chunk.
    #[test]
    fn test_take_slice_checks_chunk_take_support() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let no_take_chunk = MaskedArray::try_new(
            PrimitiveArray::from_iter(0i32..32).into_array(),
            Validity::AllValid,
            &mut ctx,
        )?
        .into_array();
        let primitive_chunk =
            PrimitiveArray::new(Buffer::from_iter(32i32..64), Validity::AllValid).into_array();
        let dtype = no_take_chunk.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![no_take_chunk, primitive_chunk], dtype)?.into_array();
        let sparse_indices = buffer![1u32, 5].into_array();

        let overlaps_no_take = SliceArray::try_new(chunked.clone(), 4..28)?;
        assert!(
            Slice::take(overlaps_no_take.as_view(), &sparse_indices, &mut ctx)?.is_none(),
            "slice overlapping a chunk without a take kernel should keep the slice"
        );

        let spans_both = SliceArray::try_new(chunked.clone(), 20..52)?;
        assert!(
            Slice::take(spans_both.as_view(), &sparse_indices, &mut ctx)?.is_none(),
            "slice spanning a chunk without a take kernel should keep the slice"
        );

        let only_primitive = SliceArray::try_new(chunked, 36..60)?;
        let pushed = Slice::take(only_primitive.as_view(), &sparse_indices, &mut ctx)?;
        assert!(
            pushed.is_some(),
            "slice whose overlapping chunks all support take should push down"
        );
        assert_arrays_eq!(
            pushed.vortex_expect("pushed down take"),
            PrimitiveArray::new(buffer![37i32, 41], Validity::AllValid).into_array(),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_take_slice_conformance() -> VortexResult<()> {
        let sliced = SliceArray::try_new(child(), 5..60)?.into_array();
        test_take_conformance(&sliced, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
