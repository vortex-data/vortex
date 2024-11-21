use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::DictArray;

impl CompareFn for DictArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<Option<ArrayData>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = other.as_constant() {
            // Ensure the other is the same length as the dictionary
            return compare(
                self.values(),
                ConstantArray::new(const_scalar, self.values().len()),
                operator,
            )
            .and_then(|values| Self::try_new(self.codes(), values))
            .map(|a| a.into_array())
            .map(Some);
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}
