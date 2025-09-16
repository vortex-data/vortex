// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::ToCanonical;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_scalar::PValue;

use crate::BitPackedArray;
use crate::bitpacking::compress::unpack_single_primitive;

impl<T: NativePType> ArrayAccessor<T> for BitPackedArray
where
    T: BitPacking + TryFrom<PValue, Error = vortex_error::VortexError> + Copy,
{
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a T>>) -> R,
    {
        // For BitPacked arrays, we need to handle potential complexity with patches
        // If there are patches, it's more efficient to canonicalize
        // This is similar to the strategy used in the take kernel
        if self.patches().is_some() {
            // Fall back to canonicalization when patches are present
            let canonical = self.to_primitive();
            return canonical.with_iterator(f);
        }

        // Collect unpacked values into a vector to provide stable references
        let mut unpacked_values: Vec<Option<T>> = Vec::with_capacity(self.len());

        let bit_width = self.bit_width() as usize;
        let packed_slice = self.packed_slice::<T>();

        // Unpack each value individually
        for i in 0..self.len() {
            let index_in_encoded = i + self.offset() as usize;

            // Check validity first
            let is_valid = self.validity.is_valid(i);

            if is_valid {
                // SAFETY: index bounds are checked by the loop and packed_slice construction
                let unpacked_value = unsafe {
                    unpack_single_primitive(packed_slice, bit_width, index_in_encoded)
                };
                unpacked_values.push(Some(unpacked_value));
            } else {
                unpacked_values.push(None);
            }
        }

        // Create iterator over references to the unpacked values
        let mut ref_iter = unpacked_values.iter().map(|opt_val| opt_val.as_ref());
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
    fn test_bitpacked_accessor() {
        // Create a test array: [1, 2, 3, 4] with 2-bit width
        let array = PrimitiveArray::new(
            buffer![1u32, 2, 3, 4],
            Validity::NonNullable,
        );

        let bitpacked = BitPackedArray::encode(array.as_ref(), 2).unwrap();

        // The accessor should yield the original values
        let mut result = Vec::new();
        bitpacked.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(1u32), Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn test_bitpacked_accessor_with_nulls() {
        // Create a test array with nulls: [Some(1), None, Some(3)]
        let array = PrimitiveArray::new(
            buffer![1u32, 0, 3], // 0 will be ignored due to validity
            Validity::from_iter([true, false, true].into_iter()),
        );

        let bitpacked = BitPackedArray::encode(array.as_ref(), 2).unwrap();

        // The accessor should preserve nulls
        let mut result = Vec::new();
        bitpacked.with_iterator(|iter| {
            for val in iter {
                result.push(val.copied());
            }
        }).unwrap();

        assert_eq!(result, vec![Some(1u32), None, Some(3)]);
    }
}