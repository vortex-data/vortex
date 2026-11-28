// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-pruning filter pushdown for [`ZstdArray`].
//!
//! Canonicalizing a `ZstdArray` inflates every frame it holds, so filtering it by canonicalizing
//! first pays for the rows the mask throws away. A frame is the smallest independently
//! decompressible unit, so the selected rows are grouped into the spans of frames that actually
//! contain them and the array is sliced to those spans — `SliceReduce` then drops the frames no
//! span reaches, and only the surviving frames are ever inflated.
//!
//! [`ZstdArray`]: crate::ZstdArray

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Zstd;

impl FilterKernel for Zstd {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(frames) = Zstd::frame_row_ranges(array, ctx)? else {
            return Ok(None);
        };
        Zstd::filter_selected_frames(array, &frames, mask)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Zstd;

    /// A session with the frame-pruning filter kernel registered.
    fn session() -> VortexSession {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    }

    /// `values_per_frame` values per frame, so the masks below straddle frame boundaries.
    fn compressed(
        values: &[Option<&str>],
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(VarBinViewArray, crate::ZstdArray)> {
        let nullability = if values.iter().any(Option::is_none) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let vbv = VarBinViewArray::from_iter(
            values.iter().map(|value| value.map(str::as_bytes)),
            DType::Utf8(nullability),
        );
        let compressed = Zstd::from_var_bin_view_without_dict(&vbv, 3, values_per_frame, ctx)?;
        Ok((vbv, compressed))
    }

    fn strings(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("value-number-{i:04}")).collect()
    }

    /// Every mask shape has to agree with filtering the uncompressed array, whether or not the
    /// pruning fires: the frames a mask skips must be exactly the ones holding no selected row.
    #[rstest]
    #[case::leading_cluster(|i: usize| i < 5)]
    #[case::trailing_cluster(|i: usize| i >= 55)]
    #[case::two_clusters(|i: usize| i < 5 || (30..35).contains(&i))]
    #[case::sparse_within_one_frame(|i: usize| (0..10).contains(&i) && i.is_multiple_of(3))]
    #[case::every_frame(|i: usize| i.is_multiple_of(7))]
    #[case::single_row(|i: usize| i == 41)]
    #[case::all_but_one(|i: usize| i != 41)]
    #[case::alternating(|i: usize| i.is_multiple_of(2))]
    fn test_filter_matches_uncompressed(#[case] selected: fn(usize) -> bool) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, 8, &mut ctx)?;

        let mask = vortex_mask::Mask::from_iter((0..60).map(selected));
        assert_arrays_eq!(
            compressed.into_array().filter(mask.clone())?,
            vbv.into_array().filter(mask)?,
            &mut ctx
        );
        Ok(())
    }

    /// Null rows hold no value, so they fall outside every frame's value range. They still have to
    /// survive the filter — a leading or trailing run of them must not be pruned away with the
    /// frames it sits beside.
    #[rstest]
    #[case::leading_nulls(0)]
    #[case::trailing_nulls(52)]
    #[case::interior_nulls(20)]
    fn test_filter_keeps_selected_null_rows(#[case] null_start: usize) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                (!(null_start..null_start + 8).contains(&i)).then_some(value.as_str())
            })
            .collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, 8, &mut ctx)?;

        // Select the null run plus one distant valid row, so pruning has a gap to skip.
        let mask = vortex_mask::Mask::from_iter(
            (0..60)
                .map(|i| (null_start..null_start + 8).contains(&i) || i == (null_start + 30) % 60),
        );
        assert_arrays_eq!(
            compressed.into_array().filter(mask.clone())?,
            vbv.into_array().filter(mask)?,
            &mut ctx
        );
        Ok(())
    }

    /// Pruning has to compose with an already-sliced array, whose rows are offset from the frames'.
    #[rstest]
    #[case::within_one_frame(9..15)]
    #[case::across_frames(5..40)]
    #[case::to_the_end(33..60)]
    fn test_filter_of_slice_matches_uncompressed(
        #[case] slice: std::ops::Range<usize>,
    ) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(60);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (vbv, compressed) = compressed(&borrowed, 8, &mut ctx)?;

        let sliced_len = slice.end - slice.start;
        let mask =
            vortex_mask::Mask::from_iter((0..sliced_len).map(|i| i < 2 || i + 2 >= sliced_len));
        assert_arrays_eq!(
            compressed
                .into_array()
                .slice(slice.clone())?
                .filter(mask.clone())?,
            vbv.into_array().slice(slice)?.filter(mask)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_filter_conformance_multi_frame() -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(20);
        let borrowed = values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>();
        let (_, compressed) = compressed(&borrowed, 4, &mut ctx)?;
        test_filter_conformance(&compressed.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_filter_conformance_nullable_multi_frame() -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();

        let values = strings(20);
        let borrowed = values
            .iter()
            .enumerate()
            .map(|(i, value)| (!i.is_multiple_of(5)).then_some(value.as_str()))
            .collect::<Vec<_>>();
        let (_, compressed) = compressed(&borrowed, 4, &mut ctx)?;
        test_filter_conformance(&compressed.into_array(), &mut ctx);
        Ok(())
    }
}
