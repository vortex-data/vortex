// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion methods and trait implementations of [`From`] and [`Into`] for [`PrimitiveArray`].

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_vector::VectorOps;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PrimitiveVector;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl PrimitiveArray {
    /// Attempts to create a `PrimitiveArray` from a [`PrimitiveVector`] given a [`Nullability`].
    ///
    /// # Errors
    ///
    /// Returns an error if the nullability is [`NonNullable`](Nullability::NonNullable) and there
    /// are nulls present in the vector.
    pub fn try_from_vector(
        primitive_vector: PrimitiveVector,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        // If we want to create a non-nullable array, then the vector should not have any nulls.
        vortex_ensure!(
            nullability.is_nullable() || primitive_vector.validity().all_true(),
            "tried to create a non-nullable `PrimitiveArray` from a `PrimitiveVector` that had nulls"
        );

        match_each_pvector!(primitive_vector, |v| {
            let (buffer, mask) = v.into_parts();
            debug_assert_eq!(buffer.len(), mask.len());

            let validity = Validity::from_mask(mask, nullability);

            // SAFETY: Since the buffer and the mask came from a valid vector, we know that the
            // length of the buffer and the validity are the same.
            Ok(unsafe { Self::new_unchecked(buffer, validity) })
        })
    }

    /// Create a PrimitiveArray from an iterator of `T`.
    /// NOTE: we cannot impl FromIterator trait since it conflicts with `FromIterator<T>`.
    pub fn from_option_iter<T: NativePType, I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut values = BufferMut::with_capacity(iter.size_hint().0);
        let mut validity = BitBufferMut::with_capacity(values.capacity());

        for i in iter {
            match i {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(e) => {
                    validity.append(true);
                    values.push(e);
                }
            }
        }
        Self::new(values.freeze(), Validity::from(validity.freeze()))
    }

    pub fn buffer<T: NativePType>(&self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        Buffer::from_byte_buffer(self.byte_buffer().clone())
    }

    pub fn into_buffer<T: NativePType>(self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        Buffer::from_byte_buffer(self.buffer)
    }

    /// Extract a mutable buffer from the PrimitiveArray. Attempts to do this with zero-copy
    /// if the buffer is uniquely owned, otherwise will make a copy.
    pub fn into_buffer_mut<T: NativePType>(self) -> BufferMut<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer_mut of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        self.into_buffer()
            .try_into_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from(&buffer))
    }

    /// Try to extract a mutable buffer from the PrimitiveArray with zero copy.
    pub fn try_into_buffer_mut<T: NativePType>(self) -> Result<BufferMut<T>, PrimitiveArray> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer_mut of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        let validity = self.validity().clone();
        Buffer::<T>::from_byte_buffer(self.into_byte_buffer())
            .try_into_mut()
            .map_err(|buffer| PrimitiveArray::new(buffer, validity))
    }
}

impl<T: NativePType> FromIterator<T> for PrimitiveArray {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values = BufferMut::from_iter(iter);
        PrimitiveArray::new(values, Validity::NonNullable)
    }
}

impl<T: NativePType> IntoArray for Buffer<T> {
    fn into_array(self) -> ArrayRef {
        PrimitiveArray::new(self, Validity::NonNullable).into_array()
    }
}

impl<T: NativePType> IntoArray for BufferMut<T> {
    fn into_array(self) -> ArrayRef {
        self.freeze().into_array()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_mask::MaskMut;
    use vortex_vector::primitive::PVector;

    use super::*;

    #[test]
    fn test_try_from_vector_with_nulls_nullable() {
        // Create a vector with some null values: [Some(1), None, Some(3), Some(4), None].
        let mut values = BufferMut::<i32>::with_capacity(5);
        values.extend_from_slice(&[1, 0, 3, 4, 0]);

        let mut validity = MaskMut::with_capacity(5);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        validity.append_n(true, 1);
        validity.append_n(true, 1);
        validity.append_n(false, 1);

        let pvector =
            PVector::try_new(values.freeze(), validity.freeze()).expect("Failed to create PVector");

        // This should succeed since we're allowing nulls.
        let result =
            PrimitiveArray::try_from_vector(pvector.into(), Nullability::Nullable).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.ptype(), PType::I32);
        assert!(result.is_valid(0));
        assert!(!result.is_valid(1));
        assert!(result.is_valid(2));
        assert!(result.is_valid(3));
        assert!(!result.is_valid(4));
    }

    #[test]
    fn test_try_from_vector_non_nullable_with_nulls_errors() {
        // Create a vector with null values: [Some(1), None, Some(3)].
        let mut values = BufferMut::<i32>::with_capacity(3);
        values.extend_from_slice(&[1, 0, 3]);

        let mut validity = MaskMut::with_capacity(3);
        validity.append_n(true, 1);
        validity.append_n(false, 1);
        validity.append_n(true, 1);

        let pvector =
            PVector::try_new(values.freeze(), validity.freeze()).expect("Failed to create PVector");

        // This should fail because we're trying to create a non-nullable array from data with
        // nulls.
        let result = PrimitiveArray::try_from_vector(pvector.into(), Nullability::NonNullable);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("non-nullable `PrimitiveArray`")
        );
    }
}
