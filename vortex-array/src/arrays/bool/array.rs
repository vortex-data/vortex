// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use arrow_array::BooleanArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, MutableBuffer};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::ToCanonical;
use crate::arrays::bool;
use crate::patches::Patches;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

pub trait BooleanBufferExt {
    /// Slice any full bytes from the buffer, leaving the offset < 8.
    fn shrink_offset(self) -> Self;
}

impl BooleanBufferExt for BooleanBuffer {
    fn shrink_offset(self) -> Self {
        let byte_offset = self.offset() / 8;
        let bit_offset = self.offset() % 8;
        let len = self.len();
        let buffer = self
            .into_inner()
            .slice_with_length(byte_offset, (len + bit_offset).div_ceil(8));
        BooleanBuffer::new(buffer, bit_offset, len)
    }
}

/// A boolean array that stores true/false values in a compact bit-packed format.
///
/// This mirrors the Apache Arrow Boolean array encoding, where each boolean value
/// is stored as a single bit rather than a full byte.
///
/// The data layout uses:
/// - A bit-packed buffer where each bit represents one boolean value (0 = false, 1 = true)
/// - An optional validity child array, which must be of type `Bool(NonNullable)`, where true values
///   indicate valid and false indicates null. if the i-th value is null in the validity child,
///   the i-th packed bit in the buffer may be 0 or 1, i.e. it is undefined.
/// - Bit-level slicing is supported with minimal overhead
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::BoolArray;
/// use vortex_array::IntoArray;
///
/// // Create from iterator using FromIterator impl
/// let array: BoolArray = [true, false, true, false].into_iter().collect();
///
/// // Slice the array
/// let sliced = array.slice(1..3);
/// assert_eq!(sliced.len(), 2);
///
/// // Access individual values
/// let value = array.scalar_at(0);
/// assert_eq!(value, true.into());
/// ```
#[derive(Clone, Debug)]
pub struct BoolArray {
    pub(super) dtype: DType,
    pub(super) buffer: BooleanBuffer,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

impl BoolArray {
    /// Constructs a new `BoolArray`.
    ///
    /// See [`BoolArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`BoolArray::new_unchecked`].
    pub fn try_new(
        buffer: ByteBuffer,
        offset: usize,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&buffer, offset, len, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(buffer, offset, len, validity) })
    }

    /// Creates a new [`BoolArray`] without validation from these components:
    ///
    /// * `buffer` is a raw [`ByteBuffer`] holding the packed bits.
    /// * `offset` is the number of bits in the start of the buffer that should be skipped when
    ///   looking up the i-th value.
    /// * `len` is the length of the array, which should correspond to the number of bits.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - `buffer` must contain at least `(offset + len).div_ceil(8)` bytes.
    /// - `offset` must be less than 8 (it represents the bit offset within the first byte).
    /// - If `validity` is `Validity::Array`, its length must exactly equal `len`.
    pub unsafe fn new_unchecked(
        buffer: ByteBuffer,
        offset: usize,
        len: usize,
        validity: Validity,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&buffer, offset, len, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `BoolArray` parameters");

        let buffer = BooleanBuffer::new(buffer.into_arrow_buffer(), offset, len);
        let buffer = buffer.shrink_offset();
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Validates the components that would be used to create a [`BoolArray`].
    ///
    /// This function checks all the invariants required by [`BoolArray::new_unchecked`].
    pub fn validate(
        buffer: &ByteBuffer,
        offset: usize,
        len: usize,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            offset < 8,
            "offset must be less than whole byte, was {offset} bits"
        );

        // Validate the buffer is large enough to hold all the bits
        let required_bytes = offset.saturating_add(len).div_ceil(8);
        vortex_ensure!(
            buffer.len() >= required_bytes,
            "BoolArray with offset={offset} len={len} cannot be built from buffer of size {}",
            buffer.len()
        );

        // Validate validity
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "BoolArray of size {len} cannot be built with validity of size {validity_len}"
            );
        }

        Ok(())
    }

    /// Creates a new [`BoolArray`] from a [`BooleanBuffer`] and [`Validity`] directly.
    ///
    /// # Panics
    ///
    /// Panics if the validity is [`Validity::Array`] and the length is not the same as the buffer.
    pub fn from_bool_buffer(buffer: BooleanBuffer, validity: Validity) -> Self {
        if let Some(validity_len) = validity.maybe_len() {
            assert_eq!(buffer.len(), validity_len);
        }

        // Shrink the buffer to remove any whole bytes.
        let buffer = buffer.shrink_offset();
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Create a new BoolArray from a set of indices and a length.
    ///
    /// All indices must be less than the length.
    pub fn from_indices<I: IntoIterator<Item = usize>>(
        length: usize,
        indices: I,
        validity: Validity,
    ) -> Self {
        let mut buffer = MutableBuffer::new_null(length);
        let buffer_slice = buffer.as_slice_mut();
        indices
            .into_iter()
            .for_each(|idx| arrow_buffer::bit_util::set_bit(buffer_slice, idx));
        Self::from_bool_buffer(
            BooleanBufferBuilder::new_from_buffer(buffer, length).finish(),
            validity,
        )
    }

    /// Returns the underlying [`BooleanBuffer`] of the array.
    pub fn boolean_buffer(&self) -> &BooleanBuffer {
        assert!(
            self.buffer.offset() < 8,
            "Offset must be <8, did we forget to call shrink_offset? Found {}",
            self.buffer.offset()
        );
        &self.buffer
    }

    /// Get a mutable version of this array.
    ///
    /// If the caller holds the only reference to the underlying buffer the underlying buffer is returned
    /// otherwise a copy is created.
    ///
    /// The second value of the tuple is a bit_offset of first value in first byte of the returned builder
    pub fn into_boolean_builder(self) -> (BooleanBufferBuilder, usize) {
        let offset = self.buffer.offset();
        let len = self.buffer.len();
        let arrow_buffer = self.buffer.into_inner();
        let mutable_buf = if arrow_buffer.ptr_offset() == 0 {
            arrow_buffer.into_mutable().unwrap_or_else(|b| {
                let mut buf = MutableBuffer::with_capacity(b.len());
                buf.extend_from_slice(b.as_slice());
                buf
            })
        } else {
            let mut buf = MutableBuffer::with_capacity(arrow_buffer.len());
            buf.extend_from_slice(arrow_buffer.as_slice());
            buf
        };

        (
            BooleanBufferBuilder::new_from_buffer(mutable_buf, offset + len),
            offset,
        )
    }

    pub fn to_mask(&self) -> Mask {
        self.maybe_to_mask()
            .vortex_expect("cannot convert nullable boolean array to mask")
    }

    pub fn maybe_to_mask(&self) -> Option<Mask> {
        self.all_valid()
            .then(|| Mask::from_buffer(self.boolean_buffer().clone()))
    }

    pub fn to_mask_fill_null_false(&self) -> Mask {
        if let Some(constant) = self.as_constant() {
            let bool_constant = constant.as_bool();
            if bool_constant.value().unwrap_or(false) {
                return Mask::new_true(self.len());
            } else {
                return Mask::new_false(self.len());
            }
        }
        // Extract a boolean buffer, treating null values to false
        let buffer = match self.validity_mask() {
            Mask::AllTrue(_) => self.boolean_buffer().clone(),
            Mask::AllFalse(_) => return Mask::new_false(self.len()),
            Mask::Values(validity) => validity.boolean_buffer().bitand(self.boolean_buffer()),
        };
        Mask::from_buffer(buffer)
    }

    pub fn patch(self, patches: &Patches) -> Self {
        let len = self.len();
        let offset = patches.offset();
        let indices = patches.indices().to_primitive();
        let values = patches.values().to_bool();

        let patched_validity =
            self.validity()
                .clone()
                .patch(len, offset, indices.as_ref(), values.validity());

        let (mut own_values, bit_offset) = self.into_boolean_builder();
        match_each_integer_ptype!(indices.ptype(), |I| {
            for (idx, value) in indices
                .as_slice::<I>()
                .iter()
                .zip_eq(values.boolean_buffer().iter())
            {
                #[allow(clippy::cast_possible_truncation)]
                own_values.set_bit(*idx as usize - offset + bit_offset, value);
            }
        });

        Self::from_bool_buffer(own_values.finish().slice(bit_offset, len), patched_validity)
    }
}

impl From<BooleanBuffer> for BoolArray {
    fn from(value: BooleanBuffer) -> Self {
        Self::from_bool_buffer(value, Validity::NonNullable)
    }
}

impl FromIterator<bool> for BoolArray {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from_bool_buffer(BooleanBuffer::from_iter(iter), Validity::NonNullable)
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let (buffer, nulls) = BooleanArray::from_iter(iter).into_parts();

        Self::from_bool_buffer(
            buffer,
            nulls.map(Validity::from).unwrap_or(Validity::AllValid),
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
    use vortex_buffer::buffer;

    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{Array, IntoArray, ToCanonical};

    #[test]
    fn bool_array() {
        let arr = BoolArray::from_iter([true, false, true]);
        let scalar = bool::try_from(&arr.scalar_at(0)).unwrap();
        assert!(scalar);
    }

    #[test]
    fn test_all_some_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(false)]);

        assert!(matches!(arr.validity(), Validity::AllValid));

        let scalar = bool::try_from(&arr.scalar_at(0)).unwrap();
        assert!(scalar);
        let scalar = bool::try_from(&arr.scalar_at(1)).unwrap();
        assert!(!scalar);
    }

    #[test]
    fn test_bool_from_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);

        let scalar = bool::try_from(&arr.scalar_at(0)).unwrap();
        assert!(scalar);

        let scalar = bool::try_from(&arr.scalar_at(1)).unwrap();
        assert!(scalar);

        let scalar = arr.scalar_at(2);
        assert!(scalar.is_null());

        let scalar = bool::try_from(&arr.scalar_at(3)).unwrap();
        assert!(!scalar);

        let scalar = arr.scalar_at(4);
        assert!(scalar.is_null());
    }

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BooleanBuffer::new_set(12));
        let sliced = arr.slice(4..12);
        let (values, offset) = sliced.to_bool().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), 12);
        assert_eq!(values.as_slice(), &[255, 15]);

        let arr = {
            let mut builder = BooleanBufferBuilder::new(12);
            builder.append(false);
            builder.append_n(11, true);
            BoolArray::from(builder.finish())
        };
        let sliced = arr.slice(4..12);
        let sliced_len = sliced.len();
        let (values, offset) = sliced.to_bool().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[254, 15]);

        // patch the underlying array
        let patches = Patches::new(
            arr.len(),
            0,
            buffer![4u32].into_array(), // This creates a non-nullable array
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(&patches);
        let arr_len = arr.len();
        let (values, offset) = arr.to_bool().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.len(), arr_len + offset);
        assert_eq!(values.as_slice(), &[238, 15]);

        // the slice should be unchanged
        let (values, offset) = sliced.to_bool().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), sliced_len + offset);
        assert_eq!(values.as_slice(), &[254, 15]); // unchanged
    }

    #[test]
    fn slice_array_in_middle() {
        let arr = BoolArray::from(BooleanBuffer::new_set(16));
        let sliced = arr.slice(4..12);
        let sliced_len = sliced.len();
        let (values, offset) = sliced.to_bool().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), sliced_len + offset);
        assert_eq!(values.as_slice(), &[255, 15]);
    }

    #[test]
    #[should_panic]
    fn patch_bools_owned() {
        let buffer = buffer![255u8; 2];
        let buf = BooleanBuffer::new(buffer.into_arrow_buffer(), 0, 15);
        let arr = BoolArray::from_bool_buffer(buf, Validity::NonNullable);
        let buf_ptr = arr.boolean_buffer().sliced().as_ptr();

        let patches = Patches::new(
            arr.len(),
            0,
            PrimitiveArray::new(buffer![0u32], Validity::AllValid).into_array(),
            BoolArray::from(BooleanBuffer::new_unset(1)).into_array(),
        );
        let arr = arr.patch(&patches);
        assert_eq!(arr.boolean_buffer().sliced().as_ptr(), buf_ptr);

        let (values, _byte_bit_offset) = arr.to_bool().into_boolean_builder();
        assert_eq!(values.as_slice(), &[254, 127]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BooleanBuffer::new_set(15));
        let sliced = arr.slice(4..15);
        let (values, offset) = sliced.to_bool().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 127]);
    }
}
