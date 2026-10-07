// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::filter::FilterArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;
use crate::compute::filter::filter_uses_direct_take;
use crate::compute::filter::filtered_run_ends;
use crate::decompress_bool::runend_decode_typed_bool;

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
/// lookup. Long boolean runs, or medium ones with a sparse selection, decode straight into the
/// selected rows; otherwise boolean runs decode every row and then filter the bits.
#[derive(Debug)]
pub(crate) struct RunEndFilteredTakeFrom;

/// Decodes selected boolean rows directly when runs average at least this many rows and the
/// selection is sparse enough (see `FUSED_BOOL_MAX_SELECTED_FRACTION_INV`).
///
/// Shorter runs make the per-run work dominate, and decoding every row before filtering the bits
/// walks the runs once with a cheap bit filter. Measured with the `run_end_dict_bool_filter`
/// benchmark, the two cross over at about 32 rows per run. A larger value favors decoding every
/// row.
const FUSED_BOOL_MIN_ROWS_PER_RUN: usize = 32;

/// Decodes selected boolean rows directly at any selectivity when runs average at least this many
/// rows.
///
/// Measured with the `run_end_dict_bool_filter` benchmark, direct decoding wins at every density
/// from 128 rows per run, while at 64 rows per run decoding every row wins once half the rows are
/// selected. A larger value favors decoding every row.
const FUSED_BOOL_ANY_SELECTION_MIN_ROWS_PER_RUN: usize = 128;

/// Below `FUSED_BOOL_ANY_SELECTION_MIN_ROWS_PER_RUN`, decodes selected boolean rows directly only
/// when fewer than one in this many rows is selected.
///
/// Measured at 64 rows per run, the two paths tie at a tenth of the rows selected. A larger value
/// favors decoding every row.
const FUSED_BOOL_MAX_SELECTED_FRACTION_INV: usize = 8;

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
        let mask = filter.filter_mask();

        let run_count = codes.ends().len();
        let decode_selected_rows = codes.len()
            >= FUSED_BOOL_ANY_SELECTION_MIN_ROWS_PER_RUN * run_count
            || (codes.len() >= FUSED_BOOL_MIN_ROWS_PER_RUN * run_count
                && mask.true_count() * FUSED_BOOL_MAX_SELECTED_FRACTION_INV < codes.len());
        if let Some(mask_values) = mask.values()
            && dict.dtype().is_boolean()
            && decode_selected_rows
        {
            // Decode the selected rows directly: each run's end moves to the number of rows
            // selected through it, and runs without selected rows become empty.
            let values = values.execute::<BoolArray>(ctx)?;
            let validity = values.validity()?.execute_mask(values.len(), ctx)?;
            let ends = codes.ends().clone().execute::<PrimitiveArray>(ctx)?;
            let filtered_ends = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
                filtered_run_ends(
                    ends.as_slice::<E>(),
                    codes.offset() as u64,
                    codes.len() as u64,
                    mask_values.bit_buffer(),
                )
            });
            return Ok(Some(runend_decode_typed_bool(
                filtered_ends.into_iter(),
                &values.to_bit_buffer(),
                validity,
                values.dtype().nullability(),
                mask_values.true_count(),
            )));
        }

        // SAFETY: we are copying ends from an existing valid RunEndArray, and the taken values
        // have one entry per run.
        let ree_array = unsafe {
            RunEnd::new_unchecked(codes.ends().clone(), values, codes.offset(), codes.len())
        }
        .into_array();

        if dict.dtype().is_boolean() {
            // Short runs or dense selections: decoding every row walks the runs once and the bit
            // filter is cheap.
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

    /// Codes with runs of 1 to 9 rows, each repeated `run_scale` times, over a 4-entry dictionary,
    /// with every seventh run null when `nullable` is set.
    fn run_codes(nullable: bool, run_scale: usize) -> PrimitiveArray {
        let codes = (0..300u32).flat_map(|run| {
            let code = (!nullable || run % 7 != 3).then_some(run % 4);
            std::iter::repeat_n(code, ((run as usize * 5) % 9 + 1) * run_scale)
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
    #[case::bool_nullable_values(BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array(), true, 2, true)]
    #[case::primitive_dense(buffer![10i64, 20, 30, 40].into_array(), false, 3, true)]
    #[case::primitive_nullable_codes(buffer![10i64, 20, 30, 40].into_array(), true, 3, true)]
    fn filtered_run_end_codes_look_up_per_run(
        #[case] values: ArrayRef,
        #[case] nullable: bool,
        #[case] keep_every: usize,
        #[case] applies: bool,
        #[values(0, 11)] offset: usize,
        // Boolean runs averaging 5 rows decode every row, 75 rows decode the selected rows only
        // for sparse masks, and 250 rows always decode the selected rows.
        #[values(1, 15, 50)] run_scale: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive_codes = run_codes(nullable, run_scale).into_array();
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
        // Long runs select enough rows per run that even the sparse mask filters runs.
        assert_eq!(looked_up.is_some(), applies || run_scale > 1);

        let expected = DictArray::try_new(primitive_codes.filter(mask)?, values)?.into_array();
        if let Some(looked_up) = looked_up {
            assert_arrays_eq!(looked_up, expected, &mut ctx);
        }
        assert_arrays_eq!(dict.into_array(), expected, &mut ctx);
        Ok(())
    }
}
