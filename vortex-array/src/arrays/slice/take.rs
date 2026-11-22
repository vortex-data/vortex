// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecute;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::slice::SliceArraySlotsExt;
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

        if !ctx
            .session()
            .kernels()
            .has_execute_parent(Dict.id(), child.encoding_id())
        {
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
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
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

    #[test]
    fn test_take_slice_conformance() -> VortexResult<()> {
        let sliced = SliceArray::try_new(child(), 5..60)?.into_array();
        test_take_conformance(&sliced, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
