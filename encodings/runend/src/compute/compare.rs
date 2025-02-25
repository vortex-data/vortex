use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareFn, Operator, compare};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_error::VortexResult;

use crate::compress::runend_decode_bools;
use crate::{RunEndArray, RunEndEncoding};

impl CompareFn<&RunEndArray> for RunEndEncoding {
    fn compare(
        &self,
        lhs: &RunEndArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            return compare(
                lhs.values(),
                &ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )
            .and_then(|values| {
                runend_decode_bools(
                    lhs.ends().to_primitive()?,
                    values.to_bool()?,
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
    use vortex_array::arrays::{BooleanBuffer, ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::{Array, ToCanonical};

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
        let res = compare(&arr, &ConstantArray::new(5, 12), Operator::Eq).unwrap();
        let res_canon = res.to_bool().unwrap();
        assert_eq!(
            res_canon.boolean_buffer(),
            &BooleanBuffer::from(vec![
                false, false, false, false, false, false, false, false, true, true, true, true
            ])
        );
    }
}
