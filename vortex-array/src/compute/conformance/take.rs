// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;

use crate::arrays::PrimitiveArray;
use crate::compute::take;
use crate::{Array, Canonical};

/// Test conformance of the take compute function for an array.
///
/// This function tests various scenarios including:
/// - Taking all elements
/// - Taking no elements
/// - Taking selective elements
/// - Taking with out-of-bounds indices (should panic)
/// - Taking with nullable indices
/// - Edge cases like empty arrays
pub fn test_take_conformance(array: &dyn Array) {
    let len = array.len();

    if len > 0 {
        test_take_all(array);
        test_take_none(array);
        test_take_selective(array);
        test_take_first_and_last(array);
        test_take_with_nullable_indices(array);
        test_take_repeated_indices(array);
    }

    test_empty_indices(array);
}

fn test_take_all(array: &dyn Array) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter(0..len as u64);
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), len);
    assert_eq!(result.dtype(), array.dtype());

    // Verify elements match
    if let Ok(orig_canonical) = array.to_canonical() {
        if let Ok(result_canonical) = result.to_canonical() {
            match (&orig_canonical, &result_canonical) {
                (Canonical::Primitive(orig_prim), Canonical::Primitive(result_prim)) => {
                    assert_eq!(orig_prim.byte_buffer(), result_prim.byte_buffer());
                }
                _ => {
                    // For non-primitive types, check scalar values
                    for i in 0..len {
                        assert_eq!(array.scalar_at(i).unwrap(), result.scalar_at(i).unwrap());
                    }
                }
            }
        }
    }
}

fn test_take_none(array: &dyn Array) {
    let indices: PrimitiveArray = PrimitiveArray::from_iter::<[u64; 0]>([]);
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

fn test_take_selective(array: &dyn Array) {
    let len = array.len();

    // Take every other element
    let indices: Vec<u64> = (0..len as u64).step_by(2).collect();
    let expected_len = indices.len();
    let indices_array = PrimitiveArray::from_iter(indices.clone());

    let result = take(array, indices_array.as_ref()).unwrap();
    assert_eq!(result.len(), expected_len);

    // Verify the taken elements
    for (result_idx, &original_idx) in indices.iter().enumerate() {
        assert_eq!(
            array.scalar_at(original_idx as usize).unwrap(),
            result.scalar_at(result_idx).unwrap()
        );
    }
}

fn test_take_first_and_last(array: &dyn Array) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter([0u64, (len - 1) as u64]);
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(array.scalar_at(0).unwrap(), result.scalar_at(0).unwrap());
    assert_eq!(
        array.scalar_at(len - 1).unwrap(),
        result.scalar_at(1).unwrap()
    );
}

fn test_take_with_nullable_indices(array: &dyn Array) {
    let len = array.len();

    // Create indices with some null values
    let indices_vec: Vec<Option<u64>> = if len >= 3 {
        vec![Some(0), None, Some((len - 1) as u64)]
    } else if len >= 2 {
        vec![Some(0), None]
    } else {
        vec![None]
    };

    let indices = PrimitiveArray::from_option_iter(indices_vec.clone());
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), indices_vec.len());
    assert_eq!(
        result.dtype(),
        &array.dtype().with_nullability(Nullability::Nullable)
    );

    // Verify values
    for (i, idx_opt) in indices_vec.iter().enumerate() {
        match idx_opt {
            Some(idx) => {
                let expected = array.scalar_at(*idx as usize).unwrap();
                let actual = result.scalar_at(i).unwrap();
                assert_eq!(expected, actual);
            }
            None => {
                assert!(result.scalar_at(i).unwrap().is_null());
            }
        }
    }
}

fn test_take_repeated_indices(array: &dyn Array) {
    if array.is_empty() {
        return;
    }

    // Take the first element multiple times
    let indices = PrimitiveArray::from_iter([0u64, 0, 0]);
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), 3);
    let first_elem = array.scalar_at(0).unwrap();
    for i in 0..3 {
        assert_eq!(result.scalar_at(i).unwrap(), first_elem);
    }
}

fn test_empty_indices(array: &dyn Array) {
    let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
    let result = take(array, indices.as_ref()).unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use super::*;
    use crate::arrays::{BoolArray, ConstantArray, NullArray, PrimitiveArray};

    #[test]
    fn test_take_primitive() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_take_bool() {
        let array = BoolArray::from_iter([true, false, true, false, true]);
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_take_nullable() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]);
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_take_constant() {
        let array = ConstantArray::new(Scalar::from(42i32), 5);
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_take_null_array() {
        let array = NullArray::new(5);
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_take_single_element() {
        let array = PrimitiveArray::from_iter([42i32]);
        test_take_conformance(array.as_ref());
    }

    #[test]
    #[should_panic]
    fn test_take_out_of_bounds() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]);
        let indices = PrimitiveArray::from_iter([0u64, 5]); // 5 is out of bounds
        let _ = take(array.as_ref(), indices.as_ref());
    }
}
