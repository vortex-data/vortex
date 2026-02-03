// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::kernel::ExecuteParentKernel;
use crate::matchers::Exact;
use crate::vtable::ValidityHelper;

#[derive(Debug)]
pub(super) struct MaskedFilterKernel;

impl ExecuteParentKernel<MaskedVTable> for MaskedFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &MaskedArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();

        // Handle trivial cases
        match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(_) => {}
        }

        // Filter the validity to get the new validity
        let filtered_validity = array.validity().filter(mask)?;

        // Filter the child array
        // The child is guaranteed to have no nulls, so filtering it is straightforward
        let filtered_child = array.child.filter(mask.clone())?;

        // Construct new MaskedArray
        let result = MaskedArray::try_new(filtered_child, filtered_validity)?.into_array();

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_filter_masked_conformance(#[case] array: MaskedArray) {
        test_filter_conformance(array.as_ref());
    }
}
