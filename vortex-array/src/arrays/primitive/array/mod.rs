// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_buffer::{Alignment, Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::{DType, NativePType, Nullability, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::ToCanonical;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

mod accessor;
mod cast;
mod conversion;
mod patch;
mod top_value;

/// A primitive array that stores [native types][vortex_dtype::NativePType] in a contiguous buffer
/// of memory, along with an optional validity child.
///
/// This mirrors the Apache Arrow Primitive layout and can be converted into and out of one
/// without allocations or copies.
///
/// The underlying buffer must be natively aligned to the primitive type they are representing.
///
/// Values are stored in their native representation with proper alignment.
/// Null values still occupy space in the buffer but are marked invalid in the validity mask.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::PrimitiveArray;
/// use vortex_array::compute::sum;
/// ///
/// // Create from iterator using FromIterator impl
/// let array: PrimitiveArray = [1i32, 2, 3, 4, 5].into_iter().collect();
///
/// // Slice the array
/// let sliced = array.slice(1..3);
///
/// // Access individual values
/// let value = sliced.scalar_at(0);
/// assert_eq!(value, 2i32.into());
///
/// // Convert into a type-erased array that can be passed to compute functions.
/// let summed = sum(sliced.as_ref()).unwrap().as_primitive().typed_value::<i64>().unwrap();
/// assert_eq!(summed, 5i64);
/// ```
#[derive(Clone, Debug)]
pub struct PrimitiveArray {
    pub(super) dtype: DType,
    pub(super) buffer: ByteBuffer,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

// TODO(connor): There are a lot of places where we could be using `new_unchecked` in the codebase.
impl PrimitiveArray {
    /// Creates a new [`PrimitiveArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`PrimitiveArray::new_unchecked`].
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        let buffer = buffer.into();
        Self::try_new(buffer, validity).vortex_expect("PrimitiveArray construction failed")
    }

    /// Constructs a new `PrimitiveArray`.
    ///
    /// See [`PrimitiveArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`PrimitiveArray::new_unchecked`].
    #[inline]
    pub fn try_new<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> VortexResult<Self> {
        Self::validate(&buffer, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(buffer, validity) })
    }

    /// Creates a new [`PrimitiveArray`] without validation from these components:
    ///
    /// * `buffer` is a typed buffer containing the primitive values.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// ## Validity Requirements
    ///
    /// - If `validity` is [`Validity::Array`], its length must exactly equal `buffer.len()`.
    #[inline]
    pub unsafe fn new_unchecked<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&buffer, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `PrimitiveArray` parameters");

        Self {
            dtype: DType::Primitive(T::PTYPE, validity.nullability()),
            buffer: buffer.into_byte_buffer(),
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`PrimitiveArray`].
    ///
    /// This function checks all the invariants required by [`PrimitiveArray::new_unchecked`].
    #[inline]
    pub fn validate<T: NativePType>(buffer: &Buffer<T>, validity: &Validity) -> VortexResult<()> {
        if let Some(len) = validity.maybe_len()
            && buffer.len() != len
        {
            return Err(vortex_err!(
                "Buffer and validity length mismatch: buffer={}, validity={}",
                buffer.len(),
                len
            ));
        }
        Ok(())
    }

    pub fn empty<T: NativePType>(nullability: Nullability) -> Self {
        Self::new(Buffer::<T>::empty(), nullability.into())
    }

    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    pub fn byte_buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    pub fn into_byte_buffer(self) -> ByteBuffer {
        self.buffer
    }

    pub fn from_byte_buffer(buffer: ByteBuffer, ptype: PType, validity: Validity) -> Self {
        match_each_native_ptype!(ptype, |T| {
            Self::new::<T>(Buffer::from_byte_buffer(buffer), validity)
        })
    }

    /// Create a PrimitiveArray from a byte buffer containing only the valid elements.
    pub fn from_values_byte_buffer(
        valid_elems_buffer: ByteBuffer,
        ptype: PType,
        validity: Validity,
        n_rows: usize,
    ) -> Self {
        let byte_width = ptype.byte_width();
        let alignment = Alignment::new(byte_width);
        let buffer = match &validity {
            Validity::AllValid | Validity::NonNullable => valid_elems_buffer.aligned(alignment),
            Validity::AllInvalid => ByteBuffer::zeroed_aligned(n_rows * byte_width, alignment),
            Validity::Array(is_valid) => {
                let bool_array = is_valid.to_bool();
                let bool_buffer = bool_array.bit_buffer();
                let mut bytes = ByteBufferMut::zeroed_aligned(n_rows * byte_width, alignment);
                for (i, valid_i) in bool_buffer.set_indices().enumerate() {
                    bytes[valid_i * byte_width..(valid_i + 1) * byte_width]
                        .copy_from_slice(&valid_elems_buffer[i * byte_width..(i + 1) * byte_width])
                }
                bytes.freeze()
            }
        };

        Self::from_byte_buffer(buffer, ptype, validity)
    }

    /// Map each element in the array to a new value.
    ///
    /// This ignores validity and maps over all maybe-null elements.
    ///
    /// TODO(ngates): we could be smarter here if validity is sparse and only run the function
    ///   over the valid elements.
    pub fn map_each<T, R, F>(self, f: F) -> PrimitiveArray
    where
        T: NativePType,
        R: NativePType,
        F: FnMut(T) -> R,
    {
        let validity = self.validity().clone();
        let buffer = match self.try_into_buffer_mut() {
            Ok(buffer_mut) => buffer_mut.map_each(f),
            Err(parray) => BufferMut::<R>::from_iter(parray.buffer::<T>().iter().copied().map(f)),
        };
        PrimitiveArray::new(buffer.freeze(), validity)
    }

    /// Map each element in the array to a new value.
    ///
    /// This doesn't ignore validity and maps over all maybe-null elements, with a bool true if
    /// valid and false otherwise.
    pub fn map_each_with_validity<T, R, F>(self, f: F) -> VortexResult<PrimitiveArray>
    where
        T: NativePType,
        R: NativePType,
        F: FnMut((T, bool)) -> R,
    {
        let validity = self.validity();

        let buf_iter = self.buffer::<T>().into_iter();

        let buffer = match &validity {
            Validity::NonNullable | Validity::AllValid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(true)).map(f))
            }
            Validity::AllInvalid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(false)).map(f))
            }
            Validity::Array(val) => {
                let val = val.to_bool();
                BufferMut::<R>::from_iter(buf_iter.zip(val.bit_buffer()).map(f))
            }
        };
        Ok(PrimitiveArray::new(buffer.freeze(), validity.clone()))
    }
}
