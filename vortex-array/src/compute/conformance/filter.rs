// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::{Array, IntoArray};
use crate::arrays::BoolArray;
use crate::compute::filter;

/// Test filter compute function with various array sizes and patterns.
/// The input array can be of any length.
pub fn test_filter(array: &dyn Array) {
    let len = array.len();
    
    // Test with arrays of any size
    if len > 0 {
        test_all_filter(array);
        test_none_filter(array);
        test_selective_filter(array);
        test_single_element_filter(array);
        test_nullable_filter(array);
    }
    
    // Test edge cases
    test_empty_array_filter(array.dtype());
    test_mismatched_lengths(array);
}

fn test_all_filter(array: &dyn Array) {
    let len = array.len();
    let mask = Mask::new_true(len);
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), len);
    
    // Verify all elements are preserved
    for i in 0..len {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_none_filter(array: &dyn Array) {
    let len = array.len();
    let mask = Mask::new_false(len);
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
}

fn test_selective_filter(array: &dyn Array) {
    let len = array.len();
    if len < 2 {
        return; // Skip for very small arrays
    }
    
    // Test alternating pattern
    let mask_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let expected_count = mask_values.iter().filter(|&&v| v).count();
    let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
    
    // Verify correct elements are kept
    for (filtered_idx, i) in (0..len).step_by(2).enumerate() {
        assert_eq!(
            filtered.scalar_at(filtered_idx).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap()
        );
    }
    
    // Test first and last only
    if len >= 2 {
        let mut mask_values = vec![false; len];
        mask_values[0] = true;
        mask_values[len - 1] = true;
        let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
        let filtered = filter(array, &mask).vortex_unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(
            filtered.scalar_at(0).vortex_unwrap(),
            array.scalar_at(0).vortex_unwrap()
        );
        assert_eq!(
            filtered.scalar_at(1).vortex_unwrap(),
            array.scalar_at(len - 1).vortex_unwrap()
        );
    }
}

fn test_single_element_filter(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }
    
    // Test selecting only the first element
    let mut mask_values = vec![false; len];
    mask_values[0] = true;
    let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(
        filtered.scalar_at(0).vortex_unwrap(),
        array.scalar_at(0).vortex_unwrap()
    );
    
    // Test selecting only the last element
    if len > 1 {
        let mut mask_values = vec![false; len];
        mask_values[len - 1] = true;
        let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
        let filtered = filter(array, &mask).vortex_unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered.scalar_at(0).vortex_unwrap(),
            array.scalar_at(len - 1).vortex_unwrap()
        );
    }
}

fn test_nullable_filter(array: &dyn Array) {
    let len = array.len();
    if len < 2 {
        return; // Skip for very small arrays
    }
    
    // Create a nullable mask where nulls are treated as false
    let bool_values: Vec<bool> = (0..len).map(|i| i % 3 == 0).collect();
    let validity_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    
    let expected_count = bool_values.iter()
        .zip(validity_values.iter())
        .filter(|(b, v)| **b && **v)
        .count();
    
    let bool_array = BoolArray::from_iter(bool_values.clone());
    let validity = crate::validity::Validity::from_iter(validity_values.clone());
    let nullable_mask = BoolArray::new(bool_array.boolean_buffer().clone(), validity);
    
    let mask = Mask::try_from(&nullable_mask).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
    
    // Verify correct elements are kept
    let mut filtered_idx = 0;
    for i in 0..len {
        if bool_values[i] && validity_values[i] {
            assert_eq!(
                filtered.scalar_at(filtered_idx).vortex_unwrap(),
                array.scalar_at(i).vortex_unwrap()
            );
            filtered_idx += 1;
        }
    }
}

fn test_empty_array_filter(dtype: &DType) {
    use crate::Canonical;
    
    let empty_array = Canonical::empty(dtype).into_array();
    let empty_mask = Mask::new_false(0);
    let filtered = filter(&empty_array, &empty_mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
    
    let empty_mask = Mask::new_true(0);
    let filtered = filter(&empty_array, &empty_mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
}

fn test_mismatched_lengths(array: &dyn Array) {
    let len = array.len();
    
    // Test mask shorter than array
    if len > 0 {
        let short_mask = Mask::new_true(len - 1);
        let result = filter(array, &short_mask);
        assert!(result.is_err(), "Filter should fail with mismatched lengths");
    }
    
    // Test mask longer than array
    let long_mask = Mask::new_true(len + 1);
    let result = filter(array, &long_mask);
    assert!(result.is_err(), "Filter should fail with mismatched lengths");
}