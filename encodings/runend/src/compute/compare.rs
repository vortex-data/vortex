use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl CompareFn<RunEndArray> for RunEndEncoding {
    fn compare(
        &self,
        lhs: &RunEndArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            return compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )
            .and_then(|values| {
                RunEndArray::with_offset_and_length(
                    lhs.ends(),
                    values,
                    lhs.validity().into_nullable(),
                    lhs.offset(),
                    lhs.len(),
                )
            })
            .map(|a| a.into_array())
            .map(Some);
        }

        // Otherwise, fall back
        Ok(None)
    }
}
#[cfg(test)]
mod test {
    use vortex_array::array::{BooleanBuffer, ConstantArray};
    use vortex_array::compute::{compare, Operator};
    use vortex_array::IntoArrayVariant;

    use crate::compute::test::ree_array;

    #[test]
    fn compare_run_end() {
        let arr = ree_array();
        let res = compare(arr, ConstantArray::new(5, 12), Operator::Eq).unwrap();
        let res_canon = res.into_bool().unwrap();
        assert_eq!(
            res_canon.boolean_buffer(),
            BooleanBuffer::from(vec![
                false, false, false, false, false, false, false, false, true, true, true, true
            ])
        );
    }
}
