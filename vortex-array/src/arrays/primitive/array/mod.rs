// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::iter;

use smallvec::smallvec;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArraySlots;
use crate::ExecutionCtx;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::validity::Validity;

mod accessor;
mod cast;
mod conversion;
mod patch;
mod top_value;

pub use patch::chunk_range;
pub use patch::patch_chunk;

use crate::ArrayRef;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::bool::BoolArrayExt;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;

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
/// use vortex_array::{VortexSessionExecute, array_session};
///
/// // Create from iterator using FromIterator impl
/// let array: PrimitiveArray = [1i32, 2, 3, 4, 5].into_iter().collect();
///
/// // Slice the array
/// let sliced = array.slice(1..3)?;
///
/// // Access individual values
/// let mut ctx = array_session().create_execution_ctx();
/// let value = sliced.execute_scalar(0, &mut ctx).unwrap();
/// assert_eq!(value, 2i32.into());
///
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub(super) ptype: PType,
    pub(super) buffer: BufferHandle,
}

impl Display for PrimitiveData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ptype: {}", self.ptype)
    }
}

pub struct PrimitiveDataParts {
    pub ptype: PType,
    pub buffer: BufferHandle,
    pub validity: Validity,
}

pub trait PrimitiveArrayExt: TypedArrayRef<Primitive> {
    fn ptype(&self) -> PType {
        match self.as_ref().dtype() {
            DType::Primitive(ptype, _) => *ptype,
            _ => unreachable!("PrimitiveArrayExt requires a primitive dtype"),
        }
    }

    fn nullability(&self) -> Nullability {
        match self.as_ref().dtype() {
            DType::Primitive(_, nullability) => *nullability,
            _ => unreachable!("PrimitiveArrayExt requires a primitive dtype"),
        }
    }

    fn validity_child(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[VALIDITY_SLOT].as_ref()
    }

    fn validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[VALIDITY_SLOT].as_ref(),
            self.nullability(),
        )
    }

    fn buffer_handle(&self) -> &BufferHandle {
        &self.buffer
    }

    fn reinterpret_cast(&self, ptype: PType) -> PrimitiveArray {
        if self.ptype() == ptype {
            return self.to_owned();
        }

        assert_eq!(
            self.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        PrimitiveArray::from_buffer_handle(self.buffer_handle().clone(), ptype, self.validity())
    }

    /// Narrow the array to the smallest possible integer type that can represent all values.
    fn narrow(&self, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        if !self.ptype().is_int() {
            return Ok(self.to_owned());
        }

        let Some(min_max) = min_max(self.as_ref(), ctx, NumericalAggregateOpts::default())? else {
            return Ok(PrimitiveArray::new(
                Buffer::<u8>::zeroed(self.len()),
                self.validity(),
            ));
        };

        // If we can't cast to i64, then leave the array as its original type.
        // It's too big to downcast anyway.
        let Ok(min) = min_max
            .min
            .cast(&PType::I64.into())
            .and_then(|s| i64::try_from(&s))
        else {
            return Ok(self.to_owned());
        };
        let Ok(max) = min_max
            .max
            .cast(&PType::I64.into())
            .and_then(|s| i64::try_from(&s))
        else {
            return Ok(self.to_owned());
        };

        let nullability = self.as_ref().dtype().nullability();

        if min < 0 || max < 0 {
            // Signed
            if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::I8, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }

            if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::I16, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }

            if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::I32, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }
        } else {
            // Unsigned
            if max <= u8::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::U8, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }

            if max <= u16::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::U16, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }

            if max <= u32::MAX as i64 {
                let result = self
                    .as_ref()
                    .cast(DType::Primitive(PType::U32, nullability))?
                    .execute::<PrimitiveArray>(ctx)?;
                return Ok(result);
            }
        }

        Ok(self.to_owned())
    }
}
impl<T: TypedArrayRef<Primitive>> PrimitiveArrayExt for T {}

// TODO(connor): There are a lot of places where we could be using `new_unchecked` in the codebase.
impl PrimitiveData {
    /// Build the slots vector for this array.
    pub(super) fn make_slots(validity: &Validity, len: usize) -> ArraySlots {
        smallvec![validity_to_child(validity, len)]
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
        _validity: Validity,
    ) -> Self {
        Self {
            ptype,
            buffer: handle,
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
    pub unsafe fn new_unchecked<T: NativePType>(buffer: Buffer<T>, _validity: Validity) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&buffer, &_validity)
            .vortex_expect("[Debug Assertion]: Invalid `PrimitiveArray` parameters");

        Self {
            ptype: T::PTYPE,
            buffer: BufferHandle::new_host(buffer.into_byte_buffer()),
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
        let dtype = DType::Primitive(T::PTYPE, nullability);
        let len = 0;
        let data = PrimitiveData::empty::<T>(nullability);
        let slots = PrimitiveData::make_slots(&Validity::from(nullability), len);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Creates a new `PrimitiveArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants.
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        let buffer = buffer.into();
        let dtype = DType::Primitive(T::PTYPE, validity.nullability());
        let len = buffer.len();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = PrimitiveData::new(buffer, validity);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Constructs a new `PrimitiveArray`.
    pub fn try_new<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> VortexResult<Self> {
        let dtype = DType::Primitive(T::PTYPE, validity.nullability());
        let len = buffer.len();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = PrimitiveData::try_new(buffer, validity)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Creates a new `PrimitiveArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`PrimitiveData::new_unchecked`].
    pub unsafe fn new_unchecked<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> Self {
        let dtype = DType::Primitive(T::PTYPE, validity.nullability());
        let len = buffer.len();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = unsafe { PrimitiveData::new_unchecked(buffer, validity) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
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
        let dtype = DType::Primitive(ptype, validity.nullability());
        let len = handle.len() / ptype.byte_width();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = unsafe { PrimitiveData::new_unchecked_from_handle(handle, ptype, validity) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Creates a new `PrimitiveArray` from a [`BufferHandle`].
    pub fn from_buffer_handle(handle: BufferHandle, ptype: PType, validity: Validity) -> Self {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let len = handle.len() / ptype.byte_width();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = PrimitiveData::from_buffer_handle(handle, ptype, validity);
        Array::try_from_parts(ArrayParts::new(Primitive, dtype, len, data).with_slots(slots))
            .vortex_expect("PrimitiveData is always valid")
    }

    /// Creates a new `PrimitiveArray` from a [`ByteBuffer`].
    pub fn from_byte_buffer(buffer: ByteBuffer, ptype: PType, validity: Validity) -> Self {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let len = buffer.len() / ptype.byte_width();
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = PrimitiveData::from_byte_buffer(buffer, ptype, validity);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Create a PrimitiveArray from a byte buffer containing only the valid elements.
    pub fn from_values_byte_buffer(
        valid_elems_buffer: ByteBuffer,
        ptype: PType,
        validity: Validity,
        n_rows: usize,
    ) -> Self {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let len = n_rows;
        let slots = PrimitiveData::make_slots(&validity, len);
        let data =
            PrimitiveData::from_values_byte_buffer(valid_elems_buffer, ptype, validity, n_rows);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Primitive, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Validates the components that would be used to create a `PrimitiveArray`.
    pub fn validate<T: NativePType>(buffer: &Buffer<T>, validity: &Validity) -> VortexResult<()> {
        PrimitiveData::validate(buffer, validity)
    }

    pub fn into_data_parts(self) -> PrimitiveDataParts {
        let validity = PrimitiveArrayExt::validity(&self);
        let ptype = PrimitiveArrayExt::ptype(&self);
        let data = self.into_data();
        PrimitiveDataParts {
            ptype,
            buffer: data.buffer,
            validity,
        }
    }

    pub fn map_each_with_validity<T, R, F>(self, f: F) -> VortexResult<Self>
    where
        T: NativePType,
        R: NativePType,
        F: FnMut((T, bool)) -> R,
    {
        let validity = PrimitiveArrayExt::validity(&self);
        let data = self.into_data();
        let buf_iter = data.to_buffer::<T>().into_iter();

        let buffer = match &validity {
            Validity::NonNullable | Validity::AllValid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(true)).map(f))
            }
            Validity::AllInvalid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(false)).map(f))
            }
            Validity::Array(val) => {
                #[expect(deprecated)]
                let val = val.to_bool().into_bit_buffer();
                BufferMut::<R>::from_iter(buf_iter.zip(val.iter()).map(f))
            }
        };
        Ok(PrimitiveArray::new(buffer.freeze(), validity))
    }
}

impl PrimitiveData {
    pub fn len(&self) -> usize {
        self.buffer.len() / self.ptype.byte_width()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn ptype(&self) -> PType {
        self.ptype
    }

    /// Get access to the buffer handle backing the array.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.buffer
    }

    pub fn from_buffer_handle(handle: BufferHandle, ptype: PType, _validity: Validity) -> Self {
        Self {
            ptype,
            buffer: handle,
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
                #[expect(deprecated)]
                let bool_array = is_valid.to_bool();
                let bool_buffer = bool_array.bit_buffer_view();
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
    ///
    /// # Panic
    /// If the buffer is not of type T this will panic
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
