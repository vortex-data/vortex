// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_array::ArrayCommon;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

use crate::unpack_iter::BitPacked;
use crate::unpack_iter::BitUnpackedChunks;

pub struct BitPackedArrayParts {
    pub offset: u16,
    pub bit_width: u8,
    pub len: usize,
    pub packed: BufferHandle,
    pub patches: Option<Patches>,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedArray {
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) common: ArrayCommon,
    pub(super) bit_width: u8,
    pub(super) packed: BufferHandle,
    pub(super) patches: Option<Patches>,
    pub(super) validity: Validity,
}

impl BitPackedArray {
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
    /// See also the [`encode`][BitPackedArrayExt::encode] method on this type for a safe path to create a new
    /// bit-packed array.
    pub(crate) unsafe fn new_unchecked(
        packed: BufferHandle,
        dtype: DType,
        validity: Validity,
        patches: Option<Patches>,
        bit_width: u8,
        len: usize,
        offset: u16,
    ) -> Self {
        Self {
            offset,
            common: ArrayCommon::new(len, dtype),
            bit_width,
            packed,
            patches,
            validity,
        }
    }
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

    // Validate offset for sliced arrays
    vortex_ensure!(
        offset < 1024,
        "Offset must be less than the full block i.e., 1024, got {offset}"
    );

    // Validate patches
    if let Some(patches) = patches {
        validate_patches(patches, ptype, length)?;
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

/// Extension trait for [`BitPackedArray`] methods.
pub trait BitPackedArrayExt: Sized {
    /// Returns the primitive type of the array.
    fn ptype(&self) -> PType;

    /// Underlying bit packed values as byte array
    fn packed(&self) -> &BufferHandle;

    /// Access the slice of packed values as an array of `T`
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T];

    /// Accessor for bit unpacked chunks
    fn unpacked_chunks<T: BitPacked>(&self) -> BitUnpackedChunks<T>;

    /// Bit-width of the packed values
    fn bit_width(&self) -> u8;

    /// Access the patches array.
    ///
    /// If present, patches MUST be a `SparseArray` with equal-length to this array, and whose
    /// indices indicate the locations of patches. The indices must have non-zero length.
    fn patches(&self) -> Option<&Patches>;

    /// Replace the patches on this array.
    fn replace_patches(&mut self, patches: Option<Patches>);

    /// Returns the offset within the first block.
    fn offset(&self) -> u16;

    /// Calculate the maximum value that **can** be contained by this array, given its bit-width.
    ///
    /// Note that this value need not actually be present in the array.
    fn max_packed_value(&self) -> usize;

    /// Decompose this array into its constituent parts.
    fn into_parts(self) -> BitPackedArrayParts;
}

impl BitPackedArrayExt for BitPackedArray {
    fn ptype(&self) -> PType {
        self.common.dtype().as_ptype()
    }

    #[inline]
    fn packed(&self) -> &BufferHandle {
        &self.packed
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed().as_host();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
        //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
        //  to get it to reinterpret the lifetime.
        unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
    }

    fn unpacked_chunks<T: BitPacked>(&self) -> BitUnpackedChunks<T> {
        assert_eq!(
            T::PTYPE,
            self.ptype(),
            "Requested type doesn't match the array ptype"
        );
        BitUnpackedChunks::new(self)
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        self.bit_width
    }

    #[inline]
    fn patches(&self) -> Option<&Patches> {
        self.patches.as_ref()
    }

    fn replace_patches(&mut self, patches: Option<Patches>) {
        self.patches = patches;
    }

    #[inline]
    fn offset(&self) -> u16 {
        self.offset
    }

    #[inline]
    fn max_packed_value(&self) -> usize {
        (1 << self.bit_width()) - 1
    }

    fn into_parts(self) -> BitPackedArrayParts {
        BitPackedArrayParts {
            offset: self.offset,
            bit_width: self.bit_width,
            len: self.common.len(),
            packed: self.packed,
            patches: self.patches,
            validity: self.validity,
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::Buffer;

    use crate::BitPackedArrayExt;
    use crate::BitPackedVTable;

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
        let packed = BitPackedVTable::encode(&uncompressed.into_array(), 1).unwrap();
        let expected = PrimitiveArray::from_option_iter(values);
        assert_arrays_eq!(packed.to_primitive(), expected);
    }

    #[test]
    fn test_encode_too_wide() {
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedVTable::encode(&uncompressed.clone().into_array(), 8)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedVTable::encode(&uncompressed.into_array(), 9)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedVTable::encode(&parray, 9).unwrap();
        assert!(packed_with_patches.patches().is_some());
        assert_arrays_eq!(
            packed_with_patches.to_primitive(),
            PrimitiveArray::new(values, vortex_array::validity::Validity::NonNullable)
        );
    }
}
