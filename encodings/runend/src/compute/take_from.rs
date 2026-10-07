// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::filter::FilterArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;
use crate::compute::filter::filter_uses_direct_take;

#[derive(Debug)]
pub(crate) struct RunEndTakeFrom;

impl ExecuteParentKernel<RunEnd> for RunEndTakeFrom {
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, RunEnd>,
        dict: ArrayView<'_, Dict>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        // Only `Primitive` and `Bool` are valid run-end value types.
        // TODO: Support additional DTypes
        if !matches!(dict.dtype(), DType::Primitive(_, _) | DType::Bool(_)) {
            return Ok(None);
        }

        // Create a new run-end array containing values as values, instead of indices as values.
        // SAFETY: we are copying ends from an existing valid RunEndArray
        let ree_array = unsafe {
            RunEnd::new_unchecked(
                array.ends().clone(),
                dict.values().take(array.values().clone())?,
                array.offset(),
                array.len(),
            )
        };
        Ok(Some(ree_array.into_array()))
    }
}

/// Looks up dictionary values per run when the run-end codes are filtered.
///
/// Rewrites `Dict(Filter(RunEnd(ends, codes)), values)` to
/// `Filter(RunEnd(ends, take(values, codes)))`. Executing the dictionary directly would decode the
/// filtered codes to one integer per selected row and then take a value per row; this takes one
/// value per run and lets the run-end filter kernel select rows. The result holds no dictionary,
/// so the dictionary filter rule cannot push the filter back into the codes.
///
/// Sparse selections, which the run-end filter kernel serves by taking rows, keep the per-row
/// lookup. Dense selections of short boolean runs decode every row and then filter the bits.
#[derive(Debug)]
pub(crate) struct RunEndFilteredTakeFrom;

/// Decodes all rows before filtering boolean values from this many selected rows per run.
///
/// Decoding every row and filtering the bits walks the runs once; filtering runs walks them for
/// the filter and again for the decode. Measured with the `run_end_dict_bool_filter` benchmark,
/// whole-row decoding wins from about one selected row per two runs. A larger value favors
/// filtering runs.
const DECODE_ALL_BOOLS_MIN_SELECTED_ROWS_PER_RUN: f64 = 0.5;

/// Filters runs instead of decoding all rows when boolean runs average at least this many rows.
///
/// With long runs, the run filter touches few runs and decodes only the selected rows, while
/// decoding all rows still pays for every row. A larger value favors decoding all rows.
const DECODE_ALL_BOOLS_MAX_ROWS_PER_RUN: usize = 256;

impl ExecuteParentKernel<Filter> for RunEndFilteredTakeFrom {
    type Parent = Dict;

    fn execute_parent(
        &self,
        filter: ArrayView<'_, Filter>,
        dict: ArrayView<'_, Dict>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        // Only `Primitive` and `Bool` are valid run-end value types.
        if !matches!(dict.dtype(), DType::Primitive(_, _) | DType::Bool(_)) {
            return Ok(None);
        }
        let Some(codes) = filter.child().as_opt::<RunEnd>() else {
            return Ok(None);
        };
        // A sparse selection takes rows directly, where decoding the selected codes is already
        // proportional to the selection and a per-run lookup only adds work.
        if filter_uses_direct_take(filter.filter_mask().true_count(), codes.ends().len()) {
            return Ok(None);
        }

        let values = dict
            .values()
            .take(codes.values().clone())?
            .execute::<Canonical>(ctx)?
            .into_array();
        let run_count = codes.ends().len();
        // SAFETY: we are copying ends from an existing valid RunEndArray, and the taken values
        // have one entry per run.
        let ree_array = unsafe {
            RunEnd::new_unchecked(codes.ends().clone(), values, codes.offset(), codes.len())
        }
        .into_array();

        let mask = filter.filter_mask();
        let selected_rows_per_run = mask.true_count() as f64 / run_count as f64;
        if dict.dtype().is_boolean()
            && codes.len() < DECODE_ALL_BOOLS_MAX_ROWS_PER_RUN * run_count
            && selected_rows_per_run >= DECODE_ALL_BOOLS_MIN_SELECTED_ROWS_PER_RUN
        {
            // Decode eagerly: a lazy filter over run-end values would filter the runs instead.
            let decoded = ree_array.execute::<Canonical>(ctx)?.into_array();
            return Ok(Some(decoded.filter(mask.clone())?));
        }

        Ok(Some(ree_array.filter(mask.clone())?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::Filter;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::kernel::ExecuteParentKernel;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::RunEnd;
    use crate::RunEndArray;
    use crate::array::RunEndArraySlotsExt;
    use crate::compute::take_from::RunEndFilteredTakeFrom;
    use crate::compute::take_from::RunEndTakeFrom;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Build a DictArray whose codes are run-end encoded.
    ///
    /// Input: `[2, 2, 2, 3, 3, 2, 2]`
    /// Dict values: `[2, 3]`
    /// Codes:       `[0, 0, 0, 1, 1, 0, 0]`
    /// RunEnd encoded codes: ends=`[3, 5, 7]`, values=`[0, 1, 0]`
    fn make_dict_with_runend_codes(ctx: &mut ExecutionCtx) -> (RunEndArray, DictArray) {
        let codes = RunEnd::encode(buffer![0u32, 0, 0, 1, 1, 0, 0].into_array(), ctx).unwrap();
        let values = buffer![2i32, 3].into_array();
        let dict = DictArray::try_new(codes.clone().into_array(), values).unwrap();
        (codes, dict)
    }

    #[test]
    fn test_execute_parent_no_offset() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (codes, dict) = make_dict_with_runend_codes(&mut ctx);

        let result = RunEndTakeFrom
            .execute_parent(codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([2i32, 2, 2, 3, 3, 2, 2]);
        let canonical = result.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(canonical, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_execute_parent_with_offset() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (codes, dict) = make_dict_with_runend_codes(&mut ctx);
        // Slice codes to positions 2..5 → logical codes [0, 1, 1] → values [2, 3, 3]
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().clone(),
                codes.values().clone(),
                2, // offset
                3, // len
            )
        };

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([2i32, 3, 3]);
        let canonical = result.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(canonical, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_execute_parent_offset_at_run_boundary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (codes, dict) = make_dict_with_runend_codes(&mut ctx);
        // Slice codes to positions 3..7 → logical codes [1, 1, 0, 0] → values [3, 3, 2, 2]
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().clone(),
                codes.values().clone(),
                3, // offset at exact run boundary
                4, // len
            )
        };

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([3i32, 3, 2, 2]);
        let canonical = result.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(canonical, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_execute_parent_single_element_offset() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (codes, dict) = make_dict_with_runend_codes(&mut ctx);
        // Slice to single element at position 4 → code=1 → value=3
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().slice(1..3)?,
                codes.values().slice(1..3)?,
                4, // offset
                1, // len
            )
        };

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([3i32]);
        let canonical = result.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(canonical, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_execute_parent_returns_none_for_non_codes_child() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (codes, dict) = make_dict_with_runend_codes(&mut ctx);

        let result = RunEndTakeFrom.execute_parent(codes.as_view(), dict.as_view(), 1, &mut ctx)?;
        assert!(result.is_none());
        Ok(())
    }

    /// Codes with runs of 1 to 9 rows over a 4-entry dictionary, with every seventh run null when
    /// `nullable` is set.
    fn run_codes(nullable: bool) -> PrimitiveArray {
        let codes = (0..300u32).flat_map(|run| {
            let code = (!nullable || run % 7 != 3).then_some(run % 4);
            std::iter::repeat_n(code, (run as usize * 5) % 9 + 1)
        });
        if nullable {
            PrimitiveArray::from_option_iter(codes)
        } else {
            PrimitiveArray::from_iter(codes.map(Option::unwrap))
        }
    }

    #[rstest]
    #[case::bool_dense(BoolArray::from_iter([true, false, false, true]).into_array(), false, 2, true)]
    #[case::bool_moderate(BoolArray::from_iter([true, false, false, true]).into_array(), false, 15, true)]
    #[case::bool_sparse(BoolArray::from_iter([true, false, false, true]).into_array(), false, 97, false)]
    #[case::bool_nullable_codes(BoolArray::from_iter([true, false, false, true]).into_array(), true, 2, true)]
    #[case::primitive_dense(buffer![10i64, 20, 30, 40].into_array(), false, 3, true)]
    #[case::primitive_nullable_codes(buffer![10i64, 20, 30, 40].into_array(), true, 3, true)]
    fn filtered_run_end_codes_look_up_per_run(
        #[case] values: ArrayRef,
        #[case] nullable: bool,
        #[case] keep_every: usize,
        #[case] applies: bool,
        #[values(0, 11)] offset: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive_codes = run_codes(nullable).into_array();
        let primitive_codes = primitive_codes.slice(offset..primitive_codes.len())?;
        let codes = RunEnd::encode(primitive_codes.clone(), &mut ctx)?.into_array();
        let mask = Mask::from_iter((0..codes.len()).map(|i| i % keep_every == 0));

        let filtered = codes.filter(mask.clone())?;
        let dict = DictArray::try_new(filtered.clone(), values.clone())?;
        let looked_up = RunEndFilteredTakeFrom.execute_parent(
            filtered.as_::<Filter>(),
            dict.as_view(),
            0,
            &mut ctx,
        )?;
        assert_eq!(looked_up.is_some(), applies);

        let expected = DictArray::try_new(primitive_codes.filter(mask)?, values)?.into_array();
        if let Some(looked_up) = looked_up {
            assert_arrays_eq!(looked_up, expected, &mut ctx);
        }
        assert_arrays_eq!(dict.into_array(), expected, &mut ctx);
        Ok(())
    }
}
