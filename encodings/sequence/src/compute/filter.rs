// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{FilterKernel, FilterKernelAdapter, take};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{SequenceArray, SequenceVTable};

impl FilterKernel for SequenceVTable {
    fn filter(&self, array: &SequenceArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // Convert mask to indices and use take
        let indices = mask
            .values()
            .ok_or_else(|| vortex_error::vortex_err!("Expected mask values"))?
            .indices();
        
        take(array.as_ref(), &indices.into_array())
    }
}

register_kernel!(FilterKernelAdapter(SequenceVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::compute::conformance::filter::test_filter;
    use vortex_array::compute::conformance::binary_numeric::test_numeric;
    use vortex_dtype::Nullability;
    
    use crate::SequenceArray;
    
    #[test]
    fn test_filter_sequence_array() {
        // Test sequence: 0, 2, 4, 6, 8
        let array = SequenceArray::typed_new(0i32, 2, Nullability::NonNullable, 5).unwrap();
        test_filter(array.as_ref());
        
        // Test sequence: 100, 105, 110, 115, 120
        let array = SequenceArray::typed_new(100u64, 5, Nullability::NonNullable, 5).unwrap();
        test_filter(array.as_ref());
        
        // Test negative sequence: -10, -15, -20, -25, -30
        let array = SequenceArray::typed_new(-10i64, -5, Nullability::NonNullable, 5).unwrap();
        test_filter(array.as_ref());
    }
    
    #[test]
    fn test_numeric_sequence_array() {
        // Test binary numeric operations
        let array1 = SequenceArray::typed_new(0i32, 1, Nullability::NonNullable, 5).unwrap();
        let array2 = SequenceArray::typed_new(10i32, 2, Nullability::NonNullable, 5).unwrap();
        test_numeric(array1.as_ref(), array2.as_ref());
        
        // Test with same arrays
        test_numeric(array1.as_ref(), array1.as_ref());
    }
}