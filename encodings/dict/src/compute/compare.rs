use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayData, IntoArrayData};
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
            return compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )
            .and_then(|values| DictArray::try_new(lhs.codes(), values))
            .map(|a| a.into_array())
            .map(Some);
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}
