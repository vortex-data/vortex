use std::fmt::Debug;

pub use compress::*;
use fastlanes::BitPacking;
use vortex_array::arrays::PrimitiveVTable;
use vortex_array::builders::ArrayBuilder;
use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{Array, ArrayExt, Canonical, EncodingId, EncodingRef, vtable};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, NativePType, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::unpack_iter::{BitPacked, BitUnpackedChunks};

mod compress;
mod compute;
mod ops;
mod serde;
pub mod unpack_iter;

vtable!(BitPacked);

impl VTable for BitPackedVTable {
    type Array = BitPackedArray;
    type Encoding = BitPackedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.bitpacked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BitPackedEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct BitPackedArray {
    offset: u16,
    len: usize,
    dtype: DType,
    bit_width: u8,
    packed: ByteBuffer,
    patches: Option<Patches>,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct BitPackedEncoding;

/// NB: All non-null values in the patches array are considered patches
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
    /// See also the [`encode`][Self::encode] method on this type for a safe path to create a new
    /// bit-packed array.
    pub unsafe fn new_unchecked(
        packed: ByteBuffer,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        bit_width: u8,
        len: usize,
    ) -> VortexResult<Self> {
        // SAFETY: checked by caller.
        unsafe {
            Self::new_unchecked_with_offset(packed, ptype, validity, patches, bit_width, len, 0)
        }
    }

    /// An unsafe constructor for a `BitPackedArray` that also specifies a slicing offset.
    ///
    /// See also [`new_unchecked`][Self::new_unchecked].
    pub(crate) unsafe fn new_unchecked_with_offset(
        packed: ByteBuffer,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        bit_width: u8,
        length: usize,
        offset: u16,
    ) -> VortexResult<Self> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        if !dtype.is_int() {
            vortex_bail!(MismatchedTypes: "integer", dtype);
        }

        if bit_width > u64::BITS as u8 {
            vortex_bail!("Unsupported bit width {}", bit_width);
        }
        if offset > 1023 {
            vortex_bail!(
                "Offset must be less than full block, i.e. 1024, got {}",
                offset
            );
        }

        if let Some(ref patches) = patches {
            // Ensure that array and patches have same PType
            if !patches.dtype().eq_ignore_nullability(ptype.into()) {
                vortex_bail!(
                    "Patches DType {} does not match BitPackedArray dtype {}",
                    patches.dtype().as_nonnullable(),
                    ptype
                )
            }
        }

        // expected packed size is in bytes
        let expected_packed_size =
            (length + offset as usize).div_ceil(1024) * (128 * bit_width as usize);
        if packed.len() != expected_packed_size {
            return Err(vortex_err!(
                "Expected {} packed bytes, got {}",
                expected_packed_size,
                packed.len()
            ));
        }

        // TODO(ngates): enforce 128 byte alignment once we have a BufferBuilder that can
        //  enforce custom alignments.
        // let packed = ByteBuffer::new_with_alignment(packed, FASTLANES_ALIGNMENT);

        Ok(Self {
            offset,
            len: length,
            dtype,
            bit_width,
            packed,
            patches,
            validity,
            stats_set: Default::default(),
        })
    }

    pub fn ptype(&self) -> PType {
        self.dtype.to_ptype()
    }

    /// Underlying bit packed values as byte array
    #[inline]
    pub fn packed(&self) -> &ByteBuffer {
        &self.packed
    }

    /// Access the slice of packed values as an array of `T`
    #[inline]
    pub fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
        //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
        //  to get it to reinterpret the lifetime.
        unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
    }

    /// Accessor for bit unpacked chunks
    pub fn unpacked_chunks<T: BitPacked>(&self) -> BitUnpackedChunks<T> {
        assert_eq!(
            T::PTYPE,
            self.ptype(),
            "Requested type doesn't match the array ptype"
        );
        BitUnpackedChunks::new(self)
    }

    /// Bit width of the packed values
    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Access the patches array.
    ///
    /// If present, patches MUST be a `SparseArray` with equal-length to this array, and whose
    /// indices indicate the locations of patches. The indices must have non-zero length.
    #[inline]
    pub fn patches(&self) -> Option<&Patches> {
        self.patches.as_ref()
    }

    pub fn replace_patches(&mut self, patches: Option<Patches>) {
        self.patches = patches;
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
    // FIXME(ngates): take a PrimitiveArray
    pub fn encode(array: &dyn Array, bit_width: u8) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            bitpack_encode(parray, bit_width, None)
        } else {
            vortex_bail!("Bitpacking can only encode primitive arrays");
        }
    }

    /// Calculate the maximum value that **can** be contained by this array, given its bit-width.
    ///
    /// Note that this value need not actually be present in the array.
    #[inline]
    pub fn max_packed_value(&self) -> usize {
        (1 << self.bit_width()) - 1
    }
}

impl ValidityHelper for BitPackedArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<BitPackedVTable> for BitPackedVTable {
    fn len(array: &BitPackedArray) -> usize {
        array.len
    }

    fn dtype(array: &BitPackedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BitPackedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<BitPackedVTable> for BitPackedVTable {
    fn canonicalize(array: &BitPackedArray) -> VortexResult<Canonical> {
        unpack(array).map(Canonical::Primitive)
    }

    fn append_to_builder(
        array: &BitPackedArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.ptype(), |$T| {
            unpack_into::<$T>(
                array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
            )
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::Buffer;

    use crate::BitPackedArray;

    // #[cfg_attr(miri, ignore)]
    // #[test]
    // fn test_bitpacked_metadata() {
    //     check_metadata(
    //         "bitpacked.metadata",
    //         RkyvMetadata(BitPackedMetadata {
    //             patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
    //             validity: ValidityMetadata::AllValid,
    //             offset: u16::MAX,
    //             bit_width: u8::MAX,
    //         }),
    //     );
    // }

    #[test]
    fn test_encode() {
        let values = [Some(1), None, Some(1), None, Some(1), None, Some(u64::MAX)];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let packed = BitPackedArray::encode(uncompressed.as_ref(), 1).unwrap();
        let expected = &[1, 0, 1, 0, 1, 0, u64::MAX];
        let results = packed.to_primitive().unwrap().as_slice::<u64>().to_vec();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_encode_too_wide() {
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedArray::encode(uncompressed.as_ref(), 8)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedArray::encode(uncompressed.as_ref(), 9)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedArray::encode(&parray, 9).unwrap();
        assert!(packed_with_patches.patches().is_some());
        assert_eq!(
            packed_with_patches
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            values.as_slice()
        );
    }
}
