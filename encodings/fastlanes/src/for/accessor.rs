// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{PrimInt, WrappingAdd};
use vortex_array::accessor::ArrayAccessor;
use vortex_array::{ToCanonical};
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::PValue;

use crate::FoRArray;

impl<T: NativePType> ArrayAccessor<T> for FoRArray
where
    T: PrimInt + WrappingAdd + Copy + TryFrom<PValue, Error = vortex_error::VortexError>,
{
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a T>>) -> R,
    {
        let reference = self
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("FoR reference must be typed value");

        let encoded = self.encoded().to_primitive();

        // Collect decompressed values into a vector to provide stable references
        let mut decompressed_values: Vec<Option<T>> = Vec::with_capacity(encoded.len());

        encoded.with_iterator(|iter: &mut dyn Iterator<Item = Option<&T>>| {
            for opt_val in iter {
                match opt_val {
                    Some(val) => decompressed_values.push(Some(val.wrapping_add(&reference))),
                    None => decompressed_values.push(None),
                }
            }
        })?;

        // Create iterator over references to the decompressed values
        let mut ref_iter = decompressed_values.iter().map(|opt_val| opt_val.as_ref());
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
    fn test_for_accessor() {
        // Create a test array: [5, 6, 7, 8, 9] (min = 5)
        let array = PrimitiveArray::new(
            buffer![5i32, 6, 7, 8, 9],
            Validity::NonNullable,
        );

        let for_array = FoRArray::encode(array).unwrap();

        // The accessor should yield the original values
        let mut result = Vec::new();
        for_array.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(5i32), Some(6), Some(7), Some(8), Some(9)]);
    }

    #[test]
    fn test_for_accessor_with_nulls() {
        // Create a test array with nulls: [Some(10), None, Some(12)]
        let array = PrimitiveArray::new(
            buffer![10i32, 0, 12], // 0 will be ignored due to validity
            Validity::from_iter([true, false, true].into_iter()),
        );

        let for_array = FoRArray::encode(array).unwrap();

        // The accessor should preserve nulls
        let mut result = Vec::new();
        for_array.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(10i32), None, Some(12)]);
    }
}