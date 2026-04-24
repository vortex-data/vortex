// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

use crate::BitPackedArray;
use crate::bitpack_compress::bitpack_encode;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitUnpackedChunks;

#[array_slots(crate::BitPacked)]
pub struct BitPackedSlots {
    /// The indices of exception values that don't fit in the bit-packed representation.
    pub patch_indices: Option<ArrayRef>,
    /// The exception values that don't fit in the bit-packed representation.
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the patch indices/values.
    pub patch_chunk_offsets: Option<ArrayRef>,
    /// The validity bitmap indicating which elements are non-null.
    pub validity_child: Option<ArrayRef>,
}

pub struct BitPackedDataParts {
    pub offset: u16,
    pub bit_width: u8,
    pub len: usize,
    pub packed: BufferHandle,
    pub patches: Option<Patches>,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedData {
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) bit_width: u8,
    pub(super) packed: BufferHandle,
    /// The offset metadata from patches, needed to reconstruct Patches from slots.
    pub(super) patch_offset: Option<usize>,
    /// The offset_within_chunk metadata from patches.
    pub(super) patch_offset_within_chunk: Option<usize>,
}

impl Display for BitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "bit_width: {}, offset: {}", self.bit_width, self.offset)
    }
}

impl BitPackedData {
    /// Create a new bitpacked array using a buffer of packed data.
    ///
    /// The packed data should be interpreted as a sequence of values with size `bit_width`.
    ///
    /// # Errors
    ///
    /// This method returns errors if any of the metadata is inconsistent, for example the packed
    /// buffer provided does not have the right size according to the supplied length and target
    /// PType.
    ///
    /// # Safety
    ///
    /// For signed arrays, it is the caller's responsibility to ensure that there are no values
    /// that can be interpreted once unpacked to the provided PType.
    ///
    /// This invariant is upheld by the compressor, but callers must ensure this if they wish to
    /// construct a new `BitPackedArray` from parts.
    ///
    /// See also the [`encode`][Self::encode] method on this type for a safe path to create a new
    /// bit-packed array.
    /// A safe constructor for a `BitPackedArray` from its components:
    ///
    /// * `packed` is ByteBuffer holding the compressed data that was packed with FastLanes
    ///   bit-packing to a `bit_width` bits per value. `length` is the length of the original
    ///   vector. Note that the packed is padded with zeros to the next multiple of 1024 elements
    ///   if `length` is not divisible by 1024.
    /// * `ptype` of the original data
    /// * `validity` to track any nulls
    /// * `patches` optionally provided for values that did not pack
    ///
    /// Any failure in validation will result in an error.
    ///
    /// # Validation
    ///
    /// * The `ptype` must be an integer
    /// * `validity` must have `length` len
    /// * Any patches must have any `array_len` equal to `length`
    /// * The `packed` buffer must be exactly sized to hold `length` values of `bit_width` rounded
    ///   up to the next multiple of 1024.
    ///
    /// Any violation of these preconditions will result in an error.
    pub fn try_new(
        packed: BufferHandle,
        patches: Option<Patches>,
        bit_width: u8,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(bit_width <= 64, "Unsupported bit width {bit_width}");
        vortex_ensure!(
            offset < 1024,
            "Offset must be less than the full block i.e., 1024, got {offset}"
        );

        let (patch_offset, patch_offset_within_chunk) = match &patches {
            Some(p) => (Some(p.offset()), p.offset_within_chunk()),
            None => (None, None),
        };

        Ok(Self {
            offset,
            bit_width,
            packed,
            patch_offset,
            patch_offset_within_chunk,
        })
    }

    pub(crate) fn validate(
        packed: &BufferHandle,
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        bit_width: u8,
        length: usize,
        offset: u16,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);
        vortex_ensure!(bit_width <= 64, "Unsupported bit width {bit_width}");

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "BitPackedArray validity length {validity_len} != array length {length}",
            );
        }

        // Validate patches
        if let Some(patches) = patches {
            Self::validate_patches(patches, ptype, length)?;
        }

        // Validate packed buffer
        let expected_packed_len =
            (length + offset as usize).div_ceil(1024) * (128 * bit_width as usize);
        vortex_ensure!(
            packed.len() == expected_packed_len,
            "Expected {} packed bytes, got {}",
            expected_packed_len,
            packed.len()
        );

        Ok(())
    }

    fn validate_patches(patches: &Patches, ptype: PType, len: usize) -> VortexResult<()> {
        // Ensure that array and patches have same ptype
        vortex_ensure!(
            patches.dtype().eq_ignore_nullability(ptype.into()),
            "Patches DType {} does not match BitPackedArray dtype {}",
            patches.dtype().as_nonnullable(),
            ptype
        );

        vortex_ensure!(
            patches.array_len() == len,
            "BitPackedArray patches length {} != expected {len}",
            patches.array_len(),
        );

        Ok(())
    }

    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
    }

    /// Underlying bit packed values as byte array
    #[inline]
    pub fn packed(&self) -> &BufferHandle {
        &self.packed
    }

    /// Access the slice of packed values as an array of `T`
    #[inline]
    pub fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed().as_host();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
        //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
        //  to get it to reinterpret the lifetime.
        unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
    }

    /// Accessor for bit unpacked chunks
    pub fn unpacked_chunks<T: BitPackedIter>(
        &self,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<BitUnpackedChunks<T>> {
        assert_eq!(
            T::PTYPE,
            self.ptype(dtype),
            "Requested type doesn't match the array ptype"
        );
        BitUnpackedChunks::try_new(self, len)
    }

    /// Bit-width of the packed values
    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Bit-pack an array of primitive integers down to the target bit-width using the FastLanes
    /// SIMD-accelerated packing kernels.
    ///
    /// # Errors
    ///
    /// If the provided array is not an integer type, an error will be returned.
    ///
    /// If the provided array contains negative values, an error will be returned.
    ///
    /// If the requested bit-width for packing is larger than the array's native width, an
    /// error will be returned.
    pub fn encode(
        array: &ArrayRef,
        bit_width: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BitPackedArray> {
        let parray: PrimitiveArray = array
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|a| vortex_err!(InvalidArgument: "Bitpacking can only encode primitive arrays, got {}", a.encoding_id()))?;
        bitpack_encode(&parray, bit_width, None, ctx)
    }

    /// Calculate the maximum value that **can** be contained by this array, given its bit-width.
    ///
    /// Note that this value need not actually be present in the array.
    #[inline]
    pub fn max_packed_value(&self) -> usize {
        (1 << self.bit_width()) - 1
    }
}

pub trait BitPackedArrayExt: BitPackedArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedData::packed(self)
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        BitPackedData::bit_width(self)
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedData::offset(self)
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        match (self.patch_indices(), self.patch_values()) {
            (Some(indices), Some(values)) => {
                let patch_offset = self
                    .patch_offset
                    .vortex_expect("has patch slots but no patch_offset");
                Some(unsafe {
                    Patches::new_unchecked(
                        self.as_ref().len(),
                        patch_offset,
                        indices.clone(),
                        values.clone(),
                        self.patch_chunk_offsets().cloned(),
                        self.patch_offset_within_chunk,
                    )
                })
            }
            _ => None,
        }
    }

    #[inline]
    fn validity(&self) -> Validity {
        child_to_validity(
            &self.validity_child().cloned(),
            self.as_ref().dtype().nullability(),
        )
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        BitPackedData::packed_slice::<T>(self)
    }

    #[inline]
    fn unpacked_chunks<T: BitPackedIter>(&self) -> VortexResult<BitUnpackedChunks<T>> {
        BitPackedData::unpacked_chunks::<T>(self, self.as_ref().dtype(), self.as_ref().len())
    }
}

impl<T: TypedArrayRef<crate::BitPacked>> BitPackedArrayExt for T {}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::Buffer;

    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    #[test]
    fn test_encode() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = [
            Some(1u64),
            None,
            Some(1),
            None,
            Some(1),
            None,
            Some(u64::MAX),
        ];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let packed = BitPackedData::encode(&uncompressed.into_array(), 1, &mut ctx).unwrap();
        let expected = PrimitiveArray::from_option_iter(values);
        let packed_primitive = packed
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(packed_primitive, expected);
    }

    #[test]
    fn test_encode_too_wide() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedData::encode(&uncompressed.clone().into_array(), 8, &mut ctx)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedData::encode(&uncompressed.into_array(), 9, &mut ctx)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedData::encode(&parray, 9, &mut ctx).unwrap();
        assert!(packed_with_patches.patches().is_some());
        let packed_primitive = packed_with_patches
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            packed_primitive,
            PrimitiveArray::new(values, vortex_array::validity::Validity::NonNullable)
        );
    }
}
