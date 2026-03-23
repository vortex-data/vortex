// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_array::Array;
use vortex_array::ArrayRef;
<<<<<<< HEAD
use vortex_array::ArrayView;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
=======
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
<<<<<<< HEAD
use vortex_array::patches::Patches;
=======
use vortex_array::stats::ArrayStats;
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

use crate::unpack_iter::BitPacked;
use crate::unpack_iter::BitUnpackedChunks;

pub(super) const VALIDITY_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

pub struct BitPackedDataParts {
    pub offset: u16,
    pub bit_width: u8,
    pub len: usize,
    pub packed: BufferHandle,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) bit_width: u8,
    pub(super) packed: BufferHandle,
<<<<<<< HEAD
    /// The offset metadata from patches, needed to reconstruct Patches from slots.
    pub(super) patch_offset: Option<usize>,
    /// The offset_within_chunk metadata from patches.
    pub(super) patch_offset_within_chunk: Option<usize>,
=======
    pub(super) stats_set: ArrayStats,
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
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
    pub(crate) unsafe fn new_unchecked(
        packed: BufferHandle,
        validity: Validity,
        bit_width: u8,
        len: usize,
        offset: u16,
    ) -> Self {
        let slots = Self::make_slots(&validity, len);

        Self {
            slots,
            offset,
            bit_width,
            packed,
<<<<<<< HEAD
            patch_offset,
            patch_offset_within_chunk,
=======
            stats_set: Default::default(),
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
        }
    }

    fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        let validity_slot = validity_to_child(validity, len);
        vec![validity_slot]
    }

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
        ptype: PType,
        validity: Validity,
        bit_width: u8,
        length: usize,
        offset: u16,
    ) -> VortexResult<Self> {
        Self::validate(&packed, ptype, &validity, bit_width, length, offset)?;

        // SAFETY: all components validated above
        unsafe {
            Ok(Self::new_unchecked(
<<<<<<< HEAD
                packed, validity, patches, bit_width, length, offset,
=======
                packed, dtype, validity, bit_width, length, offset,
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
            ))
        }
    }

    pub(crate) fn validate_against_outer(&self, dtype: &DType, len: usize) -> VortexResult<()> {
        let validity = self.validity(dtype.nullability());
        let patches = self.patches(len);
        Self::validate(
            &self.packed,
            dtype.as_ptype(),
            &validity,
            patches.as_ref(),
            self.bit_width,
            len,
            self.offset,
        )
    }

    fn validate(
        packed: &BufferHandle,
        ptype: PType,
        validity: &Validity,
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

        // Validate offset for sliced arrays
        vortex_ensure!(
            offset < 1024,
            "Offset must be less than the full block i.e., 1024, got {offset}"
        );

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

<<<<<<< HEAD
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

    /// Returns the validity as a [`Mask`](vortex_mask::Mask).
    pub fn validity_mask(&self, len: usize, nullability: Nullability) -> vortex_mask::Mask {
        self.validity(nullability).to_mask(len)
    }

    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
=======
    pub fn ptype(&self) -> PType {
        self.dtype.as_ptype()
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
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

    #[inline]
    pub fn patch_indices(&self) -> Option<&ArrayRef> {
        self.slots[PATCH_INDICES_SLOT].as_ref()
    }

    #[inline]
    pub fn patch_values(&self) -> Option<&ArrayRef> {
        self.slots[PATCH_VALUES_SLOT].as_ref()
    }

    #[inline]
    pub fn patch_chunk_offsets(&self) -> Option<&ArrayRef> {
        self.slots[PATCH_CHUNK_OFFSETS_SLOT].as_ref()
    }

    #[inline]
    pub fn validity_child(&self) -> Option<&ArrayRef> {
        self.slots[VALIDITY_SLOT].as_ref()
    }

    /// Accessor for bit unpacked chunks
    pub fn unpacked_chunks<T: BitPacked>(
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

<<<<<<< HEAD
    /// Access the patches array.
    ///
    /// Reconstructs a `Patches` from the stored slots and patch metadata.
    /// If present, patches MUST be a `SparseArray` with equal-length to this array, and whose
    /// indices indicate the locations of patches. The indices must have non-zero length.
    pub fn patches(&self, len: usize) -> Option<Patches> {
        match (self.patch_indices(), self.patch_values()) {
            (Some(indices), Some(values)) => {
                let patch_offset = self
                    .patch_offset
                    .vortex_expect("has patch slots but no patch_offset");
                Some(unsafe {
                    Patches::new_unchecked(
                        len,
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

=======
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
    /// Returns the validity, reconstructed from the stored slot.
    pub fn validity(&self, nullability: Nullability) -> Validity {
        child_to_validity(&self.validity_child().cloned(), nullability)
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Calculate the maximum value that **can** be contained by this array, given its bit-width.
    ///
    /// Note that this value need not actually be present in the array.
    #[inline]
    pub fn max_packed_value(&self) -> usize {
        (1 << self.bit_width()) - 1
    }

<<<<<<< HEAD
    pub fn into_parts(self, len: usize, nullability: Nullability) -> BitPackedDataParts {
        let patches = self.patches(len);
        let validity = self.validity(nullability);
        BitPackedDataParts {
=======
    pub fn into_parts(self) -> BitPackedArrayParts {
        let validity = self.validity();
        BitPackedArrayParts {
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
            offset: self.offset,
            bit_width: self.bit_width,
            len,
            packed: self.packed,
            validity,
        }
    }
}

pub trait BitPackedArrayExt {
    fn bitpacked_data(&self) -> &BitPackedData;
    fn bitpacked_dtype(&self) -> &DType;
    fn bitpacked_len(&self) -> usize;

    #[inline]
    fn packed(&self) -> &BufferHandle {
        self.bitpacked_data().packed()
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        self.bitpacked_data().bit_width()
    }

    #[inline]
    fn offset(&self) -> u16 {
        self.bitpacked_data().offset()
    }

    #[inline]
    fn patch_indices(&self) -> Option<&ArrayRef> {
        self.bitpacked_data().patch_indices()
    }

    #[inline]
    fn patch_values(&self) -> Option<&ArrayRef> {
        self.bitpacked_data().patch_values()
    }

    #[inline]
    fn patch_chunk_offsets(&self) -> Option<&ArrayRef> {
        self.bitpacked_data().patch_chunk_offsets()
    }

    #[inline]
    fn validity_child(&self) -> Option<&ArrayRef> {
        self.bitpacked_data().validity_child()
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        self.bitpacked_data().patches(self.bitpacked_len())
    }

    #[inline]
    fn validity(&self) -> Validity {
        self.bitpacked_data()
            .validity(self.bitpacked_dtype().nullability())
    }

    #[inline]
    fn validity_mask(&self) -> vortex_mask::Mask {
        self.validity().to_mask(self.bitpacked_len())
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        self.bitpacked_data().packed_slice::<T>()
    }

    #[inline]
    fn unpacked_chunks<T: BitPacked>(&self) -> VortexResult<BitUnpackedChunks<T>> {
        self.bitpacked_data()
            .unpacked_chunks::<T>(self.bitpacked_dtype(), self.bitpacked_len())
    }
}

impl BitPackedArrayExt for Array<crate::BitPacked> {
    fn bitpacked_data(&self) -> &BitPackedData {
        self.data()
    }

    fn bitpacked_dtype(&self) -> &DType {
        self.dtype()
    }

    fn bitpacked_len(&self) -> usize {
        self.len()
    }
}

impl BitPackedArrayExt for ArrayView<'_, crate::BitPacked> {
    fn bitpacked_data(&self) -> &BitPackedData {
        self.data()
    }

    fn bitpacked_dtype(&self) -> &DType {
        self.dtype()
    }

    fn bitpacked_len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::bitpack_compress::BitPackedEncoder;

    #[test]
    fn test_encode() {
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
        let packed = BitPackedEncoder::new(&uncompressed)
            .with_bit_width(1)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        let expected = PrimitiveArray::from_option_iter(values);
        assert_arrays_eq!(packed.to_primitive(), expected);
    }

    #[test]
    fn test_encode_too_wide() {
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedEncoder::new(&uncompressed)
            .with_bit_width(8)
            .pack()
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedEncoder::new(&uncompressed)
            .with_bit_width(9)
            .pack()
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let parray = PrimitiveArray::from_iter(0i32..=512);

<<<<<<< HEAD
        let packed_with_patches = BitPackedData::encode(&parray, 9).unwrap();
        assert!(
            packed_with_patches
                .patches(packed_with_patches.len())
                .is_some()
        );
=======
        let packed_with_patches = BitPackedEncoder::new(&parray)
            .with_bit_width(9)
            .pack()
            .unwrap();
        assert!(packed_with_patches.has_patches());
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
        assert_arrays_eq!(
            packed_with_patches.into_array().unwrap().to_primitive(),
            parray
        );
    }
}
