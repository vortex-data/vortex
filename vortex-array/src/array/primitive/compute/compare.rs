use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{arrow_compare, CompareFn, Operator};
use crate::encoding::EncodingVTable;
use crate::ArrayData;

impl CompareFn for PrimitiveArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<Option<ArrayData>> {
        // If the RHS is constant, then delegate to Arrow.
        if other.is_constant() {
            return arrow_compare(self.as_ref(), other, operator).map(Some);
        }

        // If the RHS is primitive, then delegate to Arrow.
        if other.is_encoding(PrimitiveEncoding.id()) {
            let primitive = PrimitiveArray::try_from(other.clone())?;
            return arrow_compare(self.as_ref(), primitive.as_ref(), operator).map(Some);
        }

        Ok(None)
    }
}
