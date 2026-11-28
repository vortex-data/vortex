// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-pruning take pushdown for [`ZstdArray`].
//!
//! Sorted, distinct indices describe a selection mask. When they skip a frame, filter the
//! selected frame slices in row order. Takes that touch every frame or need a gather fall back
//! to the default execution path.
//!
//! [`ZstdArray`]: crate::ZstdArray

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Zstd;

impl TakeExecute for Zstd {
    fn take(
        array: ArrayView<'_, Zstd>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let len = array.as_ref().len();
        // Unique indices asking for every row leave no frame to skip; say so before mapping frames.
        if indices.len() >= len {
            return Ok(None);
        }

        let Some(spans) = Zstd::frame_row_ranges(array, ctx)? else {
            return Ok(None);
        };

        // Skipping a frame means missing every one of its rows, so a take asking for more rows than
        // the array holds outside its smallest frame cannot skip any. Testing that against the index
        // count alone rules out the dense takes — the ones with nothing to gain — before their
        // indices are even read.
        let Some(smallest) = spans.iter().map(Range::len).min() else {
            return Ok(None);
        };
        if indices.len() > len.saturating_sub(smallest) {
            return Ok(None);
        }

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        // A null index selects no row, and a repeated or out-of-order one cannot be named by a mask.
        // Both fall through to the default canonicalize-and-gather.
        if !indices
            .as_ref()
            .validity()?
            .execute_mask(indices.as_ref().len(), ctx)?
            .all_true()
        {
            return Ok(None);
        }

        let selected = match_each_integer_ptype!(indices.ptype(), |I| {
            pruning_mask(indices.as_slice::<I>(), len, &spans)
        });
        let Some(mask) = selected else {
            return Ok(None);
        };
        let dtype = array
            .dtype()
            .union_nullability(indices.dtype().nullability());
        Zstd::filter_selected_frames(array, &spans, &mask)?
            .map(|taken| taken.cast(dtype))
            .transpose()
    }
}

fn pruning_mask<I: IntegerPType>(rows: &[I], len: usize, spans: &[Range<usize>]) -> Option<Mask> {
    if !is_ascending_unique_in_bounds(rows, len) || !skips_a_frame(spans, rows) {
        return None;
    }
    Some(Mask::from_indices(len, rows.iter().map(|&row| row.as_())))
}

/// Whether any frame holds none of the requested `rows`.
///
/// `rows` is ascending and in bounds, so each span is located by two binary searches: equal
/// insertion points mean no requested row falls inside it.
fn skips_a_frame<I: IntegerPType>(spans: &[Range<usize>], rows: &[I]) -> bool {
    spans.iter().any(|span| {
        rows.partition_point(|&row| row.as_() < span.start)
            == rows.partition_point(|&row| row.as_() < span.end)
    })
}

/// Whether `indices` is strictly ascending and within `0..len`, and so names a mask.
fn is_ascending_unique_in_bounds<I: IntegerPType>(indices: &[I], len: usize) -> bool {
    indices.first().is_some_and(|row| row.to_usize().is_some())
        && indices
            .last()
            .and_then(|row| row.to_usize())
            .is_some_and(|row| row < len)
        && indices.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ZstdArray;

    /// A session with the frame-pruning take kernel registered.
    fn session() -> VortexSession {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    }

    fn strings(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("value-number-{i:04}")).collect()
    }

    /// Six values per frame, so the index lists below start and end inside frames holding more.
    fn compressed(
        values: &[Option<&str>],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(VarBinViewArray, ZstdArray)> {
        let nullability = if values.iter().any(Option::is_none) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let vbv = VarBinViewArray::from_iter(
            values.iter().map(|value| value.map(str::as_bytes)),
            DType::Utf8(nullability),
        );
        let compressed = Zstd::from_var_bin_view_without_dict(&vbv, 3, 6, ctx)?;
        Ok((vbv, compressed))
    }

    fn index_array(indices: Vec<u64>) -> ArrayRef {
        PrimitiveArray::new(Buffer::<u64>::from(indices), Validity::NonNullable).into_array()
    }

    /// Every index shape has to agree with taking from the uncompressed array, whether the pruning
    /// fires or is declined: the frames a take skips must be exactly those holding none of its rows.
    #[rstest]
    // Prunes: the rows cluster into the leading frames.
    #[case::leading_cluster(vec![0, 1, 2, 3])]
    // Prunes: the rows cluster into two separate frames.
    #[case::two_clusters(vec![1, 2, 40, 41])]
    // Prunes: a lone row.
    #[case::single_row(vec![37])]
    // Declines the mask route (every frame holds a row) but must still be correct.
    #[case::every_frame(vec![0, 7, 13, 19, 25, 31, 37, 43, 49, 55])]
    // Declines: out of order.
    #[case::out_of_order(vec![40, 2, 17])]
    // Declines: repeated rows, which no mask can name.
    #[case::repeated(vec![3, 3, 3, 41])]
    // Declines: dense, ruled out on the index count alone.
    #[case::dense((0..58).collect())]
    fn test_take_matches_uncompressed(#[case] indices: Vec<u64>) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, &mut ctx)?;

        assert_arrays_eq!(
            compressed.into_array().take(index_array(indices.clone()))?,
            vbv.into_array().take(index_array(indices))?,
            &mut ctx
        );
        Ok(())
    }

    /// The indices are read in their own width, so every integer type has to prune the same way.
    #[rstest]
    #[case::nullable(PrimitiveArray::from_option_iter([Some(1u32), Some(40)]))]
    #[case::u8(PrimitiveArray::from_iter([1u8, 2, 40, 41]))]
    #[case::u16(PrimitiveArray::from_iter([1u16, 2, 40, 41]))]
    #[case::u32(PrimitiveArray::from_iter([1u32, 2, 40, 41]))]
    #[case::i32(PrimitiveArray::from_iter([1i32, 2, 40, 41]))]
    #[case::i64(PrimitiveArray::from_iter([1i64, 2, 40, 41]))]
    fn test_take_index_widths(#[case] indices: PrimitiveArray) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, &mut ctx)?;

        let indices = indices.into_array();
        assert_arrays_eq!(
            compressed.into_array().take(indices.clone())?,
            vbv.into_array().take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_negative_index_is_out_of_bounds() {
        assert!(!is_ascending_unique_in_bounds(&[-1i32], 60));
        assert!(!is_ascending_unique_in_bounds(&[1i64, i64::MIN], 60));
        assert!(is_ascending_unique_in_bounds(&[1i32, 59], 60));
    }

    /// Null rows hold no value and so sit outside every frame's value range, and a null *index*
    /// produces a null output. Both have to survive a take that prunes frames around them.
    #[rstest]
    #[case::leading_nulls(0)]
    #[case::trailing_nulls(52)]
    #[case::interior_nulls(20)]
    fn test_take_with_null_rows(#[case] null_start: usize) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                (!(null_start..null_start + 6).contains(&i)).then_some(value.as_str())
            })
            .collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, &mut ctx)?;

        // The null run plus one distant row, so pruning has a gap to skip.
        let indices: Vec<u64> = (null_start..null_start + 6)
            .chain(std::iter::once((null_start + 30) % 60))
            .map(|row| row as u64)
            .collect();
        assert_arrays_eq!(
            compressed.into_array().take(index_array(indices.clone()))?,
            vbv.into_array().take(index_array(indices))?,
            &mut ctx
        );
        Ok(())
    }

    /// Pruning has to compose with an already-sliced array, whose rows are offset from the frames'.
    #[rstest]
    #[case::within_one_frame(9..15)]
    #[case::across_frames(5..40)]
    #[case::to_the_end(33..60)]
    fn test_take_of_slice_matches_uncompressed(#[case] slice: Range<usize>) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, &mut ctx)?;

        let sliced_len = slice.end - slice.start;
        let indices = index_array(vec![0, 1, (sliced_len - 1) as u64]);
        assert_arrays_eq!(
            compressed
                .into_array()
                .slice(slice.clone())?
                .take(indices.clone())?,
            vbv.into_array().slice(slice)?.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_conformance_multi_frame() -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(20);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (_, compressed) = compressed(&borrowed, &mut ctx)?;
        test_take_conformance(&compressed.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_conformance_nullable_multi_frame() -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(20);
        let borrowed = values
            .iter()
            .enumerate()
            .map(|(i, value)| (!i.is_multiple_of(5)).then_some(value.as_str()))
            .collect::<Vec<_>>();
        let (_, compressed) = compressed(&borrowed, &mut ctx)?;
        test_take_conformance(&compressed.into_array(), &mut ctx);
        Ok(())
    }
}
