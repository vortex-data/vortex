// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::accessor::ArrayAccessor;
use vortex_array::{IntoArray, ToCanonical};
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_scalar::PValue;

use crate::RunEndArray;

impl<T: NativePType> ArrayAccessor<T> for RunEndArray
where
    T: TryFrom<PValue, Error = vortex_error::VortexError> + Copy,
{
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a T>>) -> R,
    {
        // For RunEnd arrays, we can efficiently decode by reading each run once
        // and expanding it to the logical positions
        let mut decoded_values: Vec<Option<T>> = Vec::with_capacity(self.len());

        // Get the canonical form of the values array to read from it
        let values_canonical = self.values().to_canonical();

        match values_canonical {
            vortex_array::Canonical::Primitive(values_array) => {
                // Use ArrayAccessor on the primitive values array
                values_array.with_iterator(|values_iter| {
                    let values_vec: Vec<Option<&T>> = values_iter.collect();

                    // For each logical position, find the physical run index and get the value
                    for logical_idx in 0..self.len() {
                        let physical_idx = self.find_physical_index(logical_idx);

                        // Get the value from the physical index in the values array
                        if let Some(value_ref) = values_vec.get(physical_idx) {
                            decoded_values.push(value_ref.copied());
                        } else {
                            decoded_values.push(None);
                        }
                    }
                })?;
            }
            vortex_array::Canonical::Bool(_bool_array) => {
                // For boolean RunEnd arrays with non-boolean target type T,
                // fall back to canonicalization of the entire array
                let canonical = self.to_primitive();
                return canonical.with_iterator(f);
            }
            _ => {
                // For other types, fall back to canonicalization of the entire array
                let canonical = self.to_primitive();
                return canonical.with_iterator(f);
            }
        }

        // Create iterator over references to the decoded values
        let mut ref_iter = decoded_values.iter().map(|opt_val| opt_val.as_ref());
        Ok(f(&mut ref_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    #[test]
    fn test_runend_accessor() {
        // Create a test array: [1, 1, 2, 2, 2, 3] (runs: [2, 5, 6], values: [1, 2, 3])
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3],
            Validity::NonNullable,
        );

        let runend = RunEndArray::encode(arr.into_array()).unwrap();

        // The accessor should yield the original values
        let mut result = Vec::new();
        runend.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(1i32), Some(1), Some(2), Some(2), Some(2), Some(3)]);
    }

    #[test]
    fn test_runend_accessor_with_nulls() {
        // Create a test array with nulls: [Some(1), Some(1), None, None, Some(3)]
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 0, 0, 3], // 0s will be ignored due to validity
            Validity::from_iter([true, true, false, false, true].into_iter()),
        );

        let runend = RunEndArray::encode(arr.into_array()).unwrap();

        // The accessor should preserve nulls
        let mut result = Vec::new();
        runend.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(1i32), Some(1), None, None, Some(3)]);
    }
}