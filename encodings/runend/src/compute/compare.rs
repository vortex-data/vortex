// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::CompareKernel;
use vortex_array::compute::CompareKernelAdapter;
use vortex_array::compute::Operator;
use vortex_array::compute::compare;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;
use crate::compress::runend_decode_bools;

impl CompareKernel for RunEndVTable {
    fn compare(
        &self,
        lhs: &RunEndArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            let values = compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()).as_ref(),
                operator,
            )?;
            let decoded = runend_decode_bools(
                lhs.ends().to_primitive(),
                values.to_bool(),
                lhs.offset(),
                lhs.len(),
            )?;
            return Ok(Some(decoded.into_array()));
        }

        // Otherwise, fall back
        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(RunEndVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn compare_run_end() {
        let arr = ree_array();
        let res = compare(
            arr.as_ref(),
            ConstantArray::new(5, 12).as_ref(),
            Operator::Eq,
        )
        .unwrap();
        let expected = BoolArray::from_iter([
            false, false, false, false, false, false, false, false, true, true, true, true,
        ]);
        assert_arrays_eq!(res, expected);
    }
}
