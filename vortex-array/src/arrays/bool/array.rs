// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::BooleanArray;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::bool;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::validity::Validity;

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
/// # fn main() -> vortex_error::VortexResult<()> {
/// use vortex_array::arrays::BoolArray;
/// use vortex_array::IntoArray;
///
/// // Create from iterator using FromIterator impl
/// let array: BoolArray = [true, false, true, false].into_iter().collect();
///
/// // Slice the array
/// let sliced = array.slice(1..3)?;
/// assert_eq!(sliced.len(), 2);
///
/// // Access individual values
/// let value = array.scalar_at(0).unwrap();
/// assert_eq!(value, true.into());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct BoolArray {
    pub(super) dtype: DType,
    pub(super) bits: BufferHandle,
    pub(super) offset: usize,
    pub(super) len: usize,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

pub struct BoolArrayParts {
    pub bits: BufferHandle,
    pub offset: usize,
    pub len: usize,
    pub validity: Validity,
}

impl BoolArray {
    /// Constructs a new `BoolArray`.
    ///
    /// # Panics
    ///
    /// Panics if the validity length is not equal to the bit buffer length.
    pub fn new(bits: BitBuffer, validity: Validity) -> Self {
        Self::try_new(bits, validity).vortex_expect("Failed to create BoolArray")
    }

    /// Constructs a new `BoolArray` from a `BufferHandle`.
    ///
    /// # Panics
    ///
    /// Panics if the validity length is not equal to the bit buffer length.
    pub fn new_handle(handle: BufferHandle, offset: usize, len: usize, validity: Validity) -> Self {
        Self::try_new_from_handle(handle, offset, len, validity)
            .vortex_expect("Failed to create BoolArray from BufferHandle")
    }

    /// Constructs a new `BoolArray`.
    ///
    /// See [`BoolArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`BoolArray::new_unchecked`].
    pub fn try_new(bits: BitBuffer, validity: Validity) -> VortexResult<Self> {
        let bits = bits.shrink_offset();
        Self::validate(&bits, &validity)?;

        let (offset, len, buffer) = bits.into_inner();

        Ok(Self {
            dtype: DType::Bool(validity.nullability()),
            bits: BufferHandle::new_host(buffer),
            offset,
            len,
            validity,
            stats_set: ArrayStats::default(),
        })
    }

    /// Build a new bool array from a `BufferHandle`, returning an error if the offset is
    /// too large or the buffer is not large enough to hold the values.
    ///
    /// # Error
    ///
    /// Error if the inputs fail validation. See also `try_new`.
    pub fn try_new_from_handle(
        bits: BufferHandle,
        offset: usize,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        vortex_ensure!(offset < 8, "BitBuffer offset must be <8, got {}", offset);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "BoolArray of size {} cannot be built with validity of size {validity_len}",
                len,
            );
        }

        vortex_ensure!(
            bits.len() * 8 >= (len + offset),
            "provided BufferHandle with offset {offset} len {len} had size {} bits",
            bits.len() * 8,
        );

        Ok(Self {
            dtype: DType::Bool(validity.nullability()),
            bits,
            offset,
            len,
            validity,
            stats_set: ArrayStats::default(),
        })
    }

    /// Creates a new [`BoolArray`] without validation from these components:
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity length is equal to the bit buffer length.
    pub unsafe fn new_unchecked(bits: BitBuffer, validity: Validity) -> Self {
        if cfg!(debug_assertions) {
            Self::new(bits, validity)
        } else {
            let (offset, len, buffer) = bits.into_inner();

            Self {
                dtype: DType::Bool(validity.nullability()),
                bits: BufferHandle::new_host(buffer),
                offset,
                len,
                validity,
                stats_set: ArrayStats::default(),
            }
        }
    }

    /// Validates the components that would be used to create a [`BoolArray`].
    ///
    /// This function checks all the invariants required by [`BoolArray::new_unchecked`].
    pub fn validate(bits: &BitBuffer, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            bits.offset() < 8,
            "BitBuffer offset must be <8, got {}",
            bits.offset()
        );

        // Validate validity
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == bits.len(),
                "BoolArray of size {} cannot be built with validity of size {validity_len}",
                bits.len()
            );
        }

        Ok(())
    }

    /// Splits into owned parts
    #[inline]
    pub fn into_parts(self) -> BoolArrayParts {
        BoolArrayParts {
            bits: self.bits,
            offset: self.offset,
            len: self.len,
            validity: self.validity,
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
        let mut buffer = BitBufferMut::new_unset(length);
        indices.into_iter().for_each(|idx| buffer.set(idx));
        Self::new(buffer.freeze(), validity)
    }

    /// Returns the underlying [`BitBuffer`] of the array.
    pub fn to_bit_buffer(&self) -> BitBuffer {
        let buffer = self.bits.as_host().clone();

        BitBuffer::new_with_offset(buffer, self.len, self.offset)
    }

    /// Returns the underlying [`BitBuffer`] of the array
    pub fn into_bit_buffer(self) -> BitBuffer {
        let buffer = self.bits.unwrap_host();

        BitBuffer::new_with_offset(buffer, self.len, self.offset)
    }

    pub fn to_mask(&self) -> Mask {
        self.maybe_to_mask()
            .vortex_expect("failed to check validity")
            .vortex_expect("cannot convert nullable boolean array to mask")
    }

    pub fn maybe_to_mask(&self) -> VortexResult<Option<Mask>> {
        Ok(self
            .all_valid()?
            .then(|| Mask::from_buffer(self.to_bit_buffer())))
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
        let buffer = match self
            .validity_mask()
            .unwrap_or_else(|_| Mask::new_true(self.len()))
        {
            Mask::AllTrue(_) => self.to_bit_buffer(),
            Mask::AllFalse(_) => return Mask::new_false(self.len()),
            Mask::Values(validity) => validity.bit_buffer() & self.to_bit_buffer(),
        };
        Mask::from_buffer(buffer)
    }
}

impl From<BitBuffer> for BoolArray {
    fn from(value: BitBuffer) -> Self {
        Self::new(value, Validity::NonNullable)
    }
}

impl FromIterator<bool> for BoolArray {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::from(BitBuffer::from_iter(iter))
    }
}

impl FromIterator<Option<bool>> for BoolArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let (buffer, nulls) = BooleanArray::from_iter(iter).into_parts();

        Self::new(
            BitBuffer::from(buffer),
            nulls
                .map(|n| Validity::from(BitBuffer::from(n.into_inner())))
                .unwrap_or(Validity::AllValid),
        )
    }
}

impl IntoArray for BitBuffer {
    fn into_array(self) -> ArrayRef {
        BoolArray::new(self, Validity::NonNullable).into_array()
    }
}

impl IntoArray for BitBufferMut {
    fn into_array(self) -> ArrayRef {
        self.freeze().into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::once;
    use std::iter::repeat_n;

    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;

    use crate::DynArray;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

    #[test]
    fn bool_array() {
        let arr = BoolArray::from_iter([true, false, true]);
        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);
    }

    #[test]
    fn test_all_some_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(false)]);

        assert!(matches!(arr.validity(), Validity::AllValid));

        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);
        let scalar = bool::try_from(&arr.scalar_at(1).unwrap()).unwrap();
        assert!(!scalar);
    }

    #[test]
    fn test_bool_from_iter() {
        let arr = BoolArray::from_iter([Some(true), Some(true), None, Some(false), None]);

        let scalar = bool::try_from(&arr.scalar_at(0).unwrap()).unwrap();
        assert!(scalar);

        let scalar = bool::try_from(&arr.scalar_at(1).unwrap()).unwrap();
        assert!(scalar);

        let scalar = arr.scalar_at(2).unwrap();
        assert!(scalar.is_null());

        let scalar = bool::try_from(&arr.scalar_at(3).unwrap()).unwrap();
        assert!(!scalar);

        let scalar = arr.scalar_at(4).unwrap();
        assert!(scalar.is_null());
    }

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BitBuffer::new_set(12));
        let sliced = arr.slice(4..12).unwrap();
        assert_arrays_eq!(sliced, BoolArray::from_iter([true; 8]));

        let arr = {
            let mut builder = BitBufferMut::new_unset(12);
            (1..12).for_each(|i| builder.set(i));
            BoolArray::from(builder.freeze())
        };
        let sliced = arr.slice(4..12).unwrap();
        let expected_slice: Vec<bool> = (4..12).map(|i| (1..12).contains(&i)).collect();
        assert_arrays_eq!(sliced, BoolArray::from_iter(expected_slice.clone()));

        // patch the underlying array at index 4 to false
        let patches = Patches::new(
            arr.len(),
            0,
            buffer![4u32].into_array(),
            BoolArray::from(BitBuffer::new_unset(1)).into_array(),
            None,
        )
        .unwrap();
        let arr = arr
            .patch(&patches, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        // After patching index 4 to false: indices 1-3 and 5-11 are true, index 0 and 4 are false
        let expected_patched: Vec<bool> = (0..12).map(|i| (1..12).contains(&i) && i != 4).collect();
        assert_arrays_eq!(arr, BoolArray::from_iter(expected_patched));

        // the slice should be unchanged (still has original values before patch)
        assert_arrays_eq!(sliced, BoolArray::from_iter(expected_slice));
    }

    #[test]
    fn slice_array_in_middle() {
        let arr = BoolArray::from(BitBuffer::new_set(16));
        let sliced = arr.slice(4..12).unwrap();
        assert_arrays_eq!(sliced, BoolArray::from_iter([true; 8]));
    }

    #[test]
    fn patch_bools_owned() {
        let arr = BoolArray::from(BitBuffer::new_set(16));
        let buf_ptr = arr.to_bit_buffer().inner().as_ptr();

        let patches = Patches::new(
            arr.len(),
            0,
            PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array(),
            BoolArray::from(BitBuffer::new_unset(1)).into_array(),
            None,
        )
        .unwrap();
        let arr = arr
            .patch(&patches, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        // Verify buffer was reused in place
        assert_eq!(arr.to_bit_buffer().inner().as_ptr(), buf_ptr);

        // After patching index 0 to false: [false, true, true, ..., true] (16 values)
        let expected: BoolArray = once(false).chain(repeat_n(true, 15)).collect();
        assert_arrays_eq!(arr, expected);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BitBuffer::new_set(15));
        let sliced = arr.slice(4..15).unwrap();
        assert_arrays_eq!(sliced, BoolArray::from_iter([true; 11]));
    }
}
