// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::BoolArray;
use crate::compute::filter;

pub fn test_filter(array: &dyn Array) {
    assert_eq!(array.len(), 5);
    test_all_filter(array);
    test_none_filter(array);
    test_selective_filter(array);
    test_nullable_filter(array);
}

fn test_all_filter(array: &dyn Array) {
    let mask = Mask::try_from(&BoolArray::from_iter([true, true, true, true, true])).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 5);
    
    for i in 0..5 {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_none_filter(array: &dyn Array) {
    let mask = Mask::try_from(&BoolArray::from_iter([false, false, false, false, false])).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
}

fn test_selective_filter(array: &dyn Array) {
    let mask = Mask::try_from(&BoolArray::from_iter([true, false, true, false, true])).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 3);
    
    assert_eq!(
        filtered.scalar_at(0).vortex_unwrap(),
        array.scalar_at(0).vortex_unwrap()
    );
    assert_eq!(
        filtered.scalar_at(1).vortex_unwrap(),
        array.scalar_at(2).vortex_unwrap()
    );
    assert_eq!(
        filtered.scalar_at(2).vortex_unwrap(),
        array.scalar_at(4).vortex_unwrap()
    );
    
    let mask = Mask::try_from(&BoolArray::from_iter([false, true, false, true, false])).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 2);
    
    assert_eq!(
        filtered.scalar_at(0).vortex_unwrap(),
        array.scalar_at(1).vortex_unwrap()
    );
    assert_eq!(
        filtered.scalar_at(1).vortex_unwrap(),
        array.scalar_at(3).vortex_unwrap()
    );
}

fn test_nullable_filter(array: &dyn Array) {
    // Create a nullable boolean array with a validity mask
    let bool_values = BoolArray::from_iter([true, false, false, true, false]);
    let validity = crate::validity::Validity::from_iter([true, false, true, true, false]);
    let nullable_mask = BoolArray::new(bool_values.boolean_buffer().clone(), validity);
    
    let mask = Mask::try_from(&nullable_mask).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    // Only indices 0 and 3 have true values with valid bits
    assert_eq!(filtered.len(), 2);
    
    assert_eq!(
        filtered.scalar_at(0).vortex_unwrap(),
        array.scalar_at(0).vortex_unwrap()
    );
    assert_eq!(
        filtered.scalar_at(1).vortex_unwrap(),
        array.scalar_at(3).vortex_unwrap()
    );
}