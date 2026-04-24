// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion methods and trait implementations of [`From`] and [`Into`] for [`PrimitiveArray`].

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::validity::Validity;

impl PrimitiveArray {
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

    /// Get a buffer in host memory holding all the values.
    ///
    /// NOTE: some values may be nonsense if the validity buffer indicates that the value is null.
    pub fn to_buffer<T: NativePType>(&self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }

        Buffer::from_byte_buffer(self.buffer_handle().to_host_sync())
    }

    /// Consume the array and get a host Buffer containing the data values.
    pub fn into_buffer<T: NativePType>(self) -> Buffer<T> {
        self.into_data().into_buffer()
    }

    /// Extract a mutable buffer from the PrimitiveArray. Attempts to do this with zero-copy
    /// if the buffer is uniquely owned, otherwise will make a copy.
    pub fn into_buffer_mut<T: NativePType>(self) -> BufferMut<T> {
        self.into_data().into_buffer_mut()
    }

    /// Try to extract a mutable buffer from the PrimitiveArray with zero copy.
    pub fn try_into_buffer_mut<T: NativePType>(self) -> Result<BufferMut<T>, Buffer<T>> {
        self.into_data().try_into_buffer_mut()
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
