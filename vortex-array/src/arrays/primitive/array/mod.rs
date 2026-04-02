// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ToCanonical;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::stats::ArrayStats;
use crate::validity::Validity;

mod accessor;
mod cast;
mod conversion;
mod patch;
mod top_value;

pub use patch::chunk_range;
pub use patch::patch_chunk;

use crate::ArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::buffer::BufferHandle;

/// The validity bitmap indicating which elements are non-null.
pub(super) const VALIDITY_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

/// A primitive array that stores [native types][crate::dtype::NativePType] in a contiguous buffer
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
/// # fn main() -> vortex_error::VortexResult<()> {
/// use vortex_array::arrays::PrimitiveArray;
///
/// // Create from iterator using FromIterator impl
/// let array: PrimitiveArray = [1i32, 2, 3, 4, 5].into_iter().collect();
///
/// // Slice the array
/// let sliced = array.slice(1..3)?;
///
/// // Access individual values
/// let value = sliced.scalar_at(0).unwrap();
/// assert_eq!(value, 2i32.into());
///
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) dtype: DType,
    pub(super) buffer: BufferHandle,
    pub(super) stats_set: ArrayStats,
}

pub struct PrimitiveArrayParts {
    pub ptype: PType,
    pub buffer: BufferHandle,
    pub validity: Validity,
}

// TODO(connor): There are a lot of places where we could be using `new_unchecked` in the codebase.
impl PrimitiveData {
    /// Build the slots vector for this array.
    pub(super) fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        vec![validity_to_child(validity, len)]
    }

    /// Create a new array from a buffer handle.
    ///
    /// # Safety
    ///
    /// Should ensure that the provided BufferHandle points at sufficiently large region of aligned
    /// memory to hold the `ptype` values.
    pub unsafe fn new_unchecked_from_handle(
        handle: BufferHandle,
        ptype: PType,
        validity: Validity,
    ) -> Self {
        let len = handle.len() / ptype.byte_width();
        let slots = Self::make_slots(&validity, len);
        let dtype = DType::Primitive(ptype, validity.nullability());
        Self {
            slots,
            buffer: handle,
            dtype,
            stats_set: ArrayStats::default(),
        }
    }

    /// Creates a new `PrimitiveArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `PrimitiveArray::new_unchecked`.
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        let buffer = buffer.into();
        Self::try_new(buffer, validity).vortex_expect("PrimitiveArray construction failed")
    }

    /// Constructs a new `PrimitiveArray`.
    ///
    /// See `PrimitiveArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// `PrimitiveArray::new_unchecked`.
    #[inline]
    pub fn try_new<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> VortexResult<Self> {
        Self::validate(&buffer, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(buffer, validity) })
    }

    /// Creates a new `PrimitiveArray` without validation from these components:
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

        let len = buffer.len();
        let slots = Self::make_slots(&validity, len);
        let dtype = DType::Primitive(T::PTYPE, validity.nullability());
        Self {
            slots,
            dtype,
            buffer: BufferHandle::new_host(buffer.into_byte_buffer()),
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a `PrimitiveArray`.
    ///
    /// This function checks all the invariants required by `PrimitiveArray::new_unchecked`.
    #[inline]
    pub fn validate<T: NativePType>(buffer: &Buffer<T>, validity: &Validity) -> VortexResult<()> {
        if let Some(len) = validity.maybe_len()
            && buffer.len() != len
        {
            return Err(vortex_err!(
                InvalidArgument:
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
}

impl Array<Primitive> {
    pub fn empty<T: NativePType>(nullability: Nullability) -> Self {
        Array::try_from_data(PrimitiveData::empty::<T>(nullability))
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Creates a new `PrimitiveArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants.
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        Array::try_from_data(PrimitiveData::new(buffer, validity))
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Constructs a new `PrimitiveArray`.
    pub fn try_new<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> VortexResult<Self> {
        Array::try_from_data(PrimitiveData::try_new(buffer, validity)?)
    }

    /// Creates a new `PrimitiveArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`PrimitiveData::new_unchecked`].
    pub unsafe fn new_unchecked<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> Self {
        Array::try_from_data(unsafe { PrimitiveData::new_unchecked(buffer, validity) })
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Create a new array from a buffer handle.
    ///
    /// # Safety
    ///
    /// See [`PrimitiveData::new_unchecked_from_handle`].
    pub unsafe fn new_unchecked_from_handle(
        handle: BufferHandle,
        ptype: PType,
        validity: Validity,
    ) -> Self {
        Array::try_from_data(unsafe {
            PrimitiveData::new_unchecked_from_handle(handle, ptype, validity)
        })
        .vortex_expect("PrimitiveData is always valid")
    }

    /// Creates a new `PrimitiveArray` from a [`BufferHandle`].
    pub fn from_buffer_handle(handle: BufferHandle, ptype: PType, validity: Validity) -> Self {
        Array::try_from_data(PrimitiveData::from_buffer_handle(handle, ptype, validity))
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Creates a new `PrimitiveArray` from a [`ByteBuffer`].
    pub fn from_byte_buffer(buffer: ByteBuffer, ptype: PType, validity: Validity) -> Self {
        Array::try_from_data(PrimitiveData::from_byte_buffer(buffer, ptype, validity))
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Create a PrimitiveArray from a byte buffer containing only the valid elements.
    pub fn from_values_byte_buffer(
        valid_elems_buffer: ByteBuffer,
        ptype: PType,
        validity: Validity,
        n_rows: usize,
    ) -> Self {
        Array::try_from_data(PrimitiveData::from_values_byte_buffer(
            valid_elems_buffer,
            ptype,
            validity,
            n_rows,
        ))
        .vortex_expect("PrimitiveData is always valid")
    }

    /// Validates the components that would be used to create a `PrimitiveArray`.
    pub fn validate<T: NativePType>(buffer: &Buffer<T>, validity: &Validity) -> VortexResult<()> {
        PrimitiveData::validate(buffer, validity)
    }
}

impl PrimitiveData {
    /// Consume the primitive array and returns its component parts.
    pub fn into_parts(self) -> PrimitiveArrayParts {
        let ptype = self.ptype();
        let validity = self.validity();
        PrimitiveArrayParts {
            ptype,
            buffer: self.buffer,
            validity,
        }
    }
}

impl PrimitiveData {
    /// Returns the dtype of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.buffer.len() / self.ptype().byte_width()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reconstructs the validity from the slot state.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> Validity {
        child_to_validity(&self.slots[VALIDITY_SLOT], self.dtype.nullability())
    }

    /// Returns the validity as a [`Mask`](vortex_mask::Mask).
    pub fn validity_mask(&self) -> vortex_mask::Mask {
        self.validity().to_mask(self.len())
    }

    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    /// Get access to the buffer handle backing the array.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.buffer
    }

    pub fn from_buffer_handle(handle: BufferHandle, ptype: PType, validity: Validity) -> Self {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let len = handle.len() / ptype.byte_width();
        let slots = Self::make_slots(&validity, len);
        Self {
            slots,
            buffer: handle,
            dtype,
            stats_set: ArrayStats::default(),
        }
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
                let bool_buffer = bool_array.to_bit_buffer();
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

    /// Map each element in the array to a new value.
    ///
    /// This ignores validity and maps over all maybe-null elements.
    ///
    /// TODO(ngates): we could be smarter here if validity is sparse and only run the function
    ///   over the valid elements.
    pub fn map_each<T, R, F>(self, f: F) -> Self
    where
        T: NativePType,
        R: NativePType,
        F: FnMut(T) -> R,
    {
        let validity = self.validity();
        let buffer = match self.try_into_buffer_mut() {
            Ok(buffer_mut) => buffer_mut.map_each_in_place(f),
            Err(buffer) => BufferMut::from_iter(buffer.iter().copied().map(f)),
        };
        PrimitiveData::new(buffer.freeze(), validity)
    }

    /// Map each element in the array to a new value.
    ///
    /// This doesn't ignore validity and maps over all maybe-null elements, with a bool true if
    /// valid and false otherwise.
    pub fn map_each_with_validity<T, R, F>(self, f: F) -> VortexResult<Self>
    where
        T: NativePType,
        R: NativePType,
        F: FnMut((T, bool)) -> R,
    {
        let validity = self.validity();

        let buf_iter = self.to_buffer::<T>().into_iter();

        let buffer = match &validity {
            Validity::NonNullable | Validity::AllValid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(true)).map(f))
            }
            Validity::AllInvalid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(false)).map(f))
            }
            Validity::Array(val) => {
                let val = val.to_bool().into_bit_buffer();
                BufferMut::<R>::from_iter(buf_iter.zip(val.iter()).map(f))
            }
        };
        Ok(PrimitiveData::new(buffer.freeze(), validity))
    }

    /// Consume the array and get a host Buffer containing the data values.
    pub fn into_buffer<T: NativePType>(self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        Buffer::from_byte_buffer(self.buffer.into_host_sync())
    }

    /// Extract a mutable buffer from the PrimitiveData. Attempts to do this with zero-copy
    /// if the buffer is uniquely owned, otherwise will make a copy.
    pub fn into_buffer_mut<T: NativePType>(self) -> BufferMut<T> {
        self.try_into_buffer_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from(&buffer))
    }

    /// Try to extract a mutable buffer from the PrimitiveData with zero copy.
    pub fn try_into_buffer_mut<T: NativePType>(self) -> Result<BufferMut<T>, Buffer<T>> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer_mut of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        let buffer = Buffer::<T>::from_byte_buffer(self.buffer.into_host_sync());
        buffer.try_into_mut()
    }
}
