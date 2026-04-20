// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;

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
        //
        Ok(Some(ree_array.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::kernel::ExecuteParentKernel;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEnd;
    use crate::RunEndArray;
    use crate::array::RunEndArrayExt;
    use crate::compute::take_from::RunEndTakeFrom;

    /// Build a DictArray whose codes are run-end encoded.
    ///
    /// Input: `[2, 2, 2, 3, 3, 2, 2]`
    /// Dict values: `[2, 3]`
    /// Codes:       `[0, 0, 0, 1, 1, 0, 0]`
    /// RunEnd encoded codes: ends=`[3, 5, 7]`, values=`[0, 1, 0]`
    fn make_dict_with_runend_codes() -> (RunEndArray, DictArray) {
        let codes = RunEnd::encode(buffer![0u32, 0, 0, 1, 1, 0, 0].into_array()).unwrap();
        let values = buffer![2i32, 3].into_array();
        let dict = DictArray::try_new(codes.clone().into_array(), values).unwrap();
        (codes, dict)
    }

    #[test]
    fn test_execute_parent_no_offset() -> VortexResult<()> {
        let (codes, dict) = make_dict_with_runend_codes();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = RunEndTakeFrom
            .execute_parent(codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([2i32, 2, 2, 3, 3, 2, 2]);
        #[expect(deprecated)]
        let canonical = result.to_canonical()?.into_array();
        assert_arrays_eq!(canonical, expected);
        Ok(())
    }

    #[test]
    fn test_execute_parent_with_offset() -> VortexResult<()> {
        let (codes, dict) = make_dict_with_runend_codes();
        // Slice codes to positions 2..5 → logical codes [0, 1, 1] → values [2, 3, 3]
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().clone(),
                codes.values().clone(),
                2, // offset
                3, // len
            )
        };
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([2i32, 3, 3]);
        #[expect(deprecated)]
        let canonical = result.to_canonical()?.into_array();
        assert_arrays_eq!(canonical, expected);
        Ok(())
    }

    #[test]
    fn test_execute_parent_offset_at_run_boundary() -> VortexResult<()> {
        let (codes, dict) = make_dict_with_runend_codes();
        // Slice codes to positions 3..7 → logical codes [1, 1, 0, 0] → values [3, 3, 2, 2]
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().clone(),
                codes.values().clone(),
                3, // offset at exact run boundary
                4, // len
            )
        };
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([3i32, 3, 2, 2]);
        #[expect(deprecated)]
        let canonical = result.to_canonical()?.into_array();
        assert_arrays_eq!(canonical, expected);
        Ok(())
    }

    #[test]
    fn test_execute_parent_single_element_offset() -> VortexResult<()> {
        let (codes, dict) = make_dict_with_runend_codes();
        // Slice to single element at position 4 → code=1 → value=3
        let sliced_codes = unsafe {
            RunEnd::new_unchecked(
                codes.ends().slice(1..3)?,
                codes.values().slice(1..3)?,
                4, // offset
                1, // len
            )
        };
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = RunEndTakeFrom
            .execute_parent(sliced_codes.as_view(), dict.as_view(), 0, &mut ctx)?
            .expect("kernel should return Some");

        let expected = PrimitiveArray::from_iter([3i32]);
        #[expect(deprecated)]
        let canonical = result.to_canonical()?.into_array();
        assert_arrays_eq!(canonical, expected);
        Ok(())
    }

    #[test]
    fn test_execute_parent_returns_none_for_non_codes_child() -> VortexResult<()> {
        let (codes, dict) = make_dict_with_runend_codes();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = RunEndTakeFrom.execute_parent(codes.as_view(), dict.as_view(), 1, &mut ctx)?;
        assert!(result.is_none());
        Ok(())
    }
}
