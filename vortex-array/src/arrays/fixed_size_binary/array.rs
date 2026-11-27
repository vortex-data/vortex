// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::smallvec;
use vortex_buffer::BitBufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::array_slots;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[array_slots(FixedSizeBinary)]
pub struct FixedSizeBinarySlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
}

/// The values buffer of a [`FixedSizeBinaryArray`](crate::arrays::FixedSizeBinaryArray).
///
/// Values are stored back to back: row `i` occupies bytes `i * byte_width..(i + 1) * byte_width`
/// of the buffer, where `byte_width` comes from [`DType::FixedSizeBinary`]. Null rows still
/// occupy `byte_width` bytes, whose contents are unspecified.
///
/// This mirrors the Apache Arrow `FixedSizeBinary` layout.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::FixedSizeBinaryArray;
/// use vortex_array::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
/// use vortex_array::validity::Validity;
/// use vortex_buffer::ByteBuffer;
///
/// let array = FixedSizeBinaryArray::new(
///     ByteBuffer::copy_from(b"abcdef"),
///     2,
///     Validity::NonNullable,
/// );
///
/// assert_eq!(array.len(), 3);
/// assert_eq!(array.byte_width(), 2);
/// assert_eq!(array.value(1), b"cd");
/// ```
#[derive(Clone, Debug)]
pub struct FixedSizeBinaryData {
    pub(super) values: BufferHandle,
}

impl Display for FixedSizeBinaryData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "values: {} bytes", self.values.len())
    }
}

/// Fixed-size binary storage and validity, as returned by
/// [`FixedSizeBinaryArray::into_data_parts`](crate::arrays::FixedSizeBinaryArray::into_data_parts).
pub struct FixedSizeBinaryDataParts {
    /// The number of bytes in each value.
    pub byte_width: u32,
    /// The values, `byte_width` bytes per row, in host or device memory.
    pub values: BufferHandle,
    /// The validity of the values, which also gives the nullability of the array.
    pub validity: Validity,
}

pub trait FixedSizeBinaryArrayExt: TypedArrayRef<FixedSizeBinary> {
    /// Returns the number of bytes in each value.
    fn byte_width(&self) -> u32 {
        match self.dtype() {
            DType::FixedSizeBinary(byte_width, _) => *byte_width,
            _ => unreachable!("FixedSizeBinaryArrayExt requires a fixed-size binary dtype"),
        }
    }

    fn nullability(&self) -> Nullability {
        match self.dtype() {
            DType::FixedSizeBinary(_, nullability) => *nullability,
            _ => unreachable!("FixedSizeBinaryArrayExt requires a fixed-size binary dtype"),
        }
    }

    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.nullability())
    }

    /// Returns the host or device buffer that stores the values.
    fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }

    /// Returns the values as a single host-resident buffer, copying it from the device if needed.
    fn values(&self) -> ByteBuffer {
        self.values.to_host_sync()
    }

    /// Returns the bytes of the value at `index`, ignoring validity.
    ///
    /// # Panics
    ///
    /// Panics if the values are not on the host or `index` is out of bounds.
    fn value(&self, index: usize) -> &[u8] {
        assert!(
            index < self.len(),
            "index {index} out of bounds for length {}",
            self.len()
        );
        let byte_width = self.byte_width() as usize;
        let start = index * byte_width;
        &self.values.as_host()[start..start + byte_width]
    }
}
impl<T: TypedArrayRef<FixedSizeBinary>> FixedSizeBinaryArrayExt for T {}

impl FixedSizeBinaryData {
    /// Build the slots vector for this array.
    pub(super) fn make_slots(validity: &Validity, len: usize) -> ArraySlots {
        smallvec![validity_to_child(validity, len)]
    }

    /// Returns the host or device buffer that stores the values.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }
}

impl Array<FixedSizeBinary> {
    /// Creates a fixed-size binary array from `values`, which hold `byte_width` bytes per row.
    ///
    /// # Panics
    ///
    /// Panics on any error from [`Self::try_new`].
    pub fn new(values: ByteBuffer, byte_width: u32, validity: Validity) -> Self {
        Self::try_new(values, byte_width, validity)
            .vortex_expect("FixedSizeBinaryArray construction failed")
    }

    /// Creates a fixed-size binary array from `values`, which hold `byte_width` bytes per row.
    ///
    /// A zero `byte_width` has no bytes to count rows by, so its length comes from an array
    /// validity, or is zero otherwise. Use [`Self::try_new_handle`] to give the length explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error if the values are not a whole number of rows, or the validity has the
    /// wrong length.
    pub fn try_new(values: ByteBuffer, byte_width: u32, validity: Validity) -> VortexResult<Self> {
        let len = match (byte_width, validity.maybe_len()) {
            (0, Some(len)) => len,
            (0, None) => 0,
            (width, _) => {
                let width = width as usize;
                vortex_ensure_eq!(
                    values.len() % width,
                    0,
                    "fixed-size binary values of {} bytes are not a whole number of {width}-byte rows",
                    values.len(),
                );
                values.len() / width
            }
        };
        Self::try_new_handle(BufferHandle::new_host(values), byte_width, len, validity)
    }

    /// Creates a fixed-size binary array of `len` rows from a host or device buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer does not hold `len * byte_width` bytes, or the validity has
    /// the wrong length or dtype.
    pub fn try_new_handle(
        values: BufferHandle,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        Array::try_from_parts(Self::fixed_size_binary_parts(
            values, byte_width, len, validity,
        ))
    }

    /// Creates a fixed-size binary array without checking its layout or validity.
    ///
    /// # Safety
    ///
    /// The buffer must hold exactly `len * byte_width` bytes, and an array-backed validity must
    /// contain one non-nullable boolean per row.
    pub unsafe fn new_unchecked_handle(
        values: BufferHandle,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> Self {
        let parts = Self::fixed_size_binary_parts(values, byte_width, len, validity);
        // SAFETY: The caller guarantees the buffer length and validity.
        unsafe { Array::from_parts_unchecked(parts) }
    }

    /// Collects values of `byte_width` bytes each into a non-nullable array.
    ///
    /// # Panics
    ///
    /// Panics if any value does not have exactly `byte_width` bytes.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<V: AsRef<[u8]>, I: IntoIterator<Item = V>>(iter: I, byte_width: u32) -> Self {
        Self::from_option_iter(iter.into_iter().map(Some), byte_width).into_non_nullable_unchecked()
    }

    /// Collects optional values of `byte_width` bytes each into a nullable array. Null rows
    /// contain zero bytes.
    ///
    /// # Panics
    ///
    /// Panics if any present value does not have exactly `byte_width` bytes.
    pub fn from_option_iter<V: AsRef<[u8]>, I: IntoIterator<Item = Option<V>>>(
        iter: I,
        byte_width: u32,
    ) -> Self {
        let width = byte_width as usize;
        let iter = iter.into_iter();
        let mut values = ByteBufferMut::with_capacity(iter.size_hint().0 * width);
        let mut validity = BitBufferMut::with_capacity(iter.size_hint().0);
        for value in iter {
            match value {
                Some(value) => {
                    let value = value.as_ref();
                    assert_eq!(
                        value.len(),
                        width,
                        "fixed-size binary value has {} bytes, expected {width}",
                        value.len()
                    );
                    values.extend_from_slice(value);
                    validity.append(true);
                }
                None => {
                    values.push_n(0, width);
                    validity.append(false);
                }
            }
        }
        let len = validity.len();
        let validity = Validity::from(validity.freeze());

        // SAFETY: Each row appended `width` bytes and one validity bit.
        unsafe {
            Self::new_unchecked_handle(
                BufferHandle::new_host(values.freeze()),
                byte_width,
                len,
                validity,
            )
        }
    }

    /// Returns the byte width, values buffer and validity of the array.
    pub fn into_data_parts(self) -> FixedSizeBinaryDataParts {
        let validity = FixedSizeBinaryArrayExt::validity(&self);
        let byte_width = FixedSizeBinaryArrayExt::byte_width(&self);
        let data = self.into_data();
        FixedSizeBinaryDataParts {
            byte_width,
            values: data.values,
            validity,
        }
    }

    /// Assembles the parts of an array. [`Array::try_from_parts`] checks the buffer length and
    /// validity before exposing the array; a caller that skips validation must establish them.
    pub(super) fn fixed_size_binary_parts(
        values: BufferHandle,
        byte_width: u32,
        len: usize,
        validity: Validity,
    ) -> ArrayParts<FixedSizeBinary> {
        let dtype = DType::FixedSizeBinary(byte_width, validity.nullability());
        let slots = FixedSizeBinaryData::make_slots(&validity, len);
        let data = FixedSizeBinaryData { values };
        ArrayParts::new(FixedSizeBinary, dtype, len, data, slots)
    }

    /// Replaces the values and validity at the patch indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the patch values have another dtype, ignoring nullability.
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        vortex_ensure!(
            self.dtype().eq_ignore_nullability(patches.values().dtype()),
            "FixedSizeBinary patch values have dtype {}, expected {}",
            patches.values().dtype(),
            self.dtype(),
        );
        let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches
            .values()
            .clone()
            .execute::<FixedSizeBinaryArray>(ctx)?;

        let len = self.len();
        let byte_width = self.byte_width();
        let validity = FixedSizeBinaryArrayExt::validity(&self).patch(
            len,
            patches.offset(),
            &patch_indices.clone().into_array(),
            &patch_values.validity()?,
            ctx,
        )?;

        let mut values = self.into_data().values.unwrap_host().into_mut();
        let patch_bytes = patch_values.values();
        // Patch indices are non-negative, so signed and unsigned indices can share a kernel.
        let indices = patch_indices.reinterpret_cast(patch_indices.ptype().to_unsigned());
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            patch_records(
                values.as_mut_slice(),
                byte_width as usize,
                indices.as_slice::<I>(),
                patches.offset(),
                patch_bytes.as_slice(),
            )
        });

        Self::try_new_handle(
            BufferHandle::new_host(values.freeze()),
            byte_width,
            len,
            validity,
        )
    }

    /// Drops the validity of an array built from all-present values.
    fn into_non_nullable_unchecked(self) -> Self {
        let len = self.len();
        let FixedSizeBinaryDataParts {
            byte_width, values, ..
        } = self.into_data_parts();
        // SAFETY: The values are unchanged and the array has no validity child.
        unsafe { Self::new_unchecked_handle(values, byte_width, len, Validity::NonNullable) }
    }
}

/// Copies record `i` of `patch_values` over the record at `indices[i] - offset` of `values`.
fn patch_records<I: IntegerPType>(
    values: &mut [u8],
    byte_width: usize,
    indices: &[I],
    offset: usize,
    patch_values: &[u8],
) {
    for (i, index) in indices.iter().enumerate() {
        let target = (index.as_() - offset) * byte_width;
        let source = i * byte_width;
        values[target..target + byte_width]
            .copy_from_slice(&patch_values[source..source + byte_width]);
    }
}

/// Returns the number of bytes that `len` rows of `byte_width` bytes occupy.
pub(super) fn values_len(len: usize, byte_width: u32) -> VortexResult<usize> {
    len.checked_mul(byte_width as usize)
        .ok_or_else(|| vortex_err!("fixed-size binary values length overflows usize"))
}
