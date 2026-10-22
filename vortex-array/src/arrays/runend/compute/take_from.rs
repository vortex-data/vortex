// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Dict;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;

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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::arrays::runend::array::RunEndArraySlotsExt;
    use crate::arrays::runend::compute::take_from::RunEndTakeFrom;
    use crate::assert_arrays_eq;
    use crate::kernel::ExecuteParentKernel;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
        assert_arrays_eq!(result, expected, &mut ctx);
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
        assert_arrays_eq!(result, expected, &mut ctx);
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
        assert_arrays_eq!(result, expected, &mut ctx);
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
        assert_arrays_eq!(result, expected, &mut ctx);
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
}
