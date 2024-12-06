use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, take, CompareFn, Operator, TakeOptions};
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl CompareFn<DictArray> for DictEncoding {
    fn compare(
        &self,
        lhs: &DictArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            // Ensure the other is the same length as the dictionary
            let compare_result = compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )?;
            return take(compare_result, lhs.codes(), TakeOptions::default()).map(Some);
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}
