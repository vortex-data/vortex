// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::BooleanArray;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::bool;
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
    pub(super) bits: BitBuffer,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

pub struct BoolArrayParts {
    pub dtype: DType,
    pub bits: BitBuffer,
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
        Ok(Self {
            dtype: DType::Bool(validity.nullability()),
            bits,
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
            Self {
                dtype: DType::Bool(validity.nullability()),
                bits,
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
            dtype: self.dtype,
            bits: self.bits,
            validity: self.validity,
        }
    }

    /// Creates a new [`BoolArray`] from a [`BitBuffer`] and [`Validity`] directly.
    ///
    /// # Panics
    ///
    /// Panics if the validity is [`Validity::Array`] and the length is not the same as the buffer.
    pub fn from_bit_buffer(buffer: BitBuffer, validity: Validity) -> Self {
        if let Some(validity_len) = validity.maybe_len() {
            assert_eq!(buffer.len(), validity_len);
        }

        // Shrink the buffer to remove any whole bytes.
        let buffer = buffer.shrink_offset();
        Self {
            dtype: DType::Bool(validity.nullability()),
            bits: buffer,
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
        let mut buffer = BitBufferMut::new_unset(length);
        indices.into_iter().for_each(|idx| buffer.set(idx));
        Self::from_bit_buffer(buffer.freeze(), validity)
    }

    /// Returns the underlying [`BitBuffer`] of the array.
    pub fn bit_buffer(&self) -> &BitBuffer {
        assert!(
            self.bits.offset() < 8,
            "Offset must be <8, did we forget to call shrink_offset? Found {}",
            self.bits.offset()
        );
        &self.bits
    }

    /// Returns the underlying [`BitBuffer`] ofthe array
    pub fn into_bit_buffer(self) -> BitBuffer {
        self.bits
    }

    pub fn to_mask(&self) -> Mask {
        self.maybe_to_mask()
            .vortex_expect("failed to check validity")
            .vortex_expect("cannot convert nullable boolean array to mask")
    }

    pub fn maybe_to_mask(&self) -> VortexResult<Option<Mask>> {
        Ok(self
            .all_valid()?
            .then(|| Mask::from_buffer(self.bit_buffer().clone())))
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
            Mask::AllTrue(_) => self.bit_buffer().clone(),
            Mask::AllFalse(_) => return Mask::new_false(self.len()),
            Mask::Values(validity) => validity.bit_buffer() & self.bit_buffer(),
        };
        Mask::from_buffer(buffer)
    }
}

impl From<BitBuffer> for BoolArray {
    fn from(value: BitBuffer) -> Self {
        Self::from_bit_buffer(value, Validity::NonNullable)
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

        Self::from_bit_buffer(
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
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;

    use crate::Array;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
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
        assert_eq!(sliced.len(), 8);
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.len(), 8);
        assert_eq!(values.as_slice(), &[255, 255]);

        let arr = {
            let mut builder = BitBufferMut::new_unset(12);
            (1..12).for_each(|i| builder.set(i));
            BoolArray::from(builder.freeze())
        };
        let sliced = arr.slice(4..12).unwrap();
        let sliced_len = sliced.len();
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.as_slice(), &[254, 15]);

        // patch the underlying array
        let patches = Patches::new(
            arr.len(),
            0,
            buffer![4u32].into_array(), // This creates a non-nullable array
            BoolArray::from(BitBuffer::new_unset(1)).into_array(),
            None,
        )
        .unwrap();
        let arr = arr.patch(&patches).unwrap();
        let arr_len = arr.len();
        let values = arr.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.len(), arr_len);
        assert_eq!(values.as_slice(), &[238, 15]);

        // the slice should be unchanged
        let sliced = sliced.to_bool();
        let values = sliced.into_bit_buffer().into_mut();
        assert_eq!(values.len(), sliced_len);
        assert_eq!(values.as_slice(), &[254, 15]); // unchanged
    }

    #[test]
    fn slice_array_in_middle() {
        let arr = BoolArray::from(BitBuffer::new_set(16));
        let sliced = arr.slice(4..12).unwrap();
        let sliced_len = sliced.len();
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.len(), sliced_len);
        assert_eq!(values.as_slice(), &[255, 255]);
    }

    #[test]
    fn patch_bools_owned() {
        let arr = BoolArray::from(BitBuffer::new_set(16));
        let buf_ptr = arr.bit_buffer().inner().as_ptr();

        let patches = Patches::new(
            arr.len(),
            0,
            PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array(),
            BoolArray::from(BitBuffer::new_unset(1)).into_array(),
            None,
        )
        .unwrap();
        let arr = arr.patch(&patches).unwrap();
        assert_eq!(arr.bit_buffer().inner().as_ptr(), buf_ptr);

        let values = arr.into_bit_buffer();
        assert_eq!(values.inner().as_slice(), &[254, 255]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BitBuffer::new_set(15));
        let sliced = arr.slice(4..15).unwrap();
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.as_slice(), &[255, 255]);
    }
}
