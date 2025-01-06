use std::fmt::{Debug, Display};
use std::sync::Arc;

use ::serde::{Deserialize, Serialize};
pub use compress::*;
use fastlanes::BitPacking;
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::stats::{StatisticsVTable, StatsSet};
use vortex_array::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use vortex_array::variants::{PrimitiveArrayTrait, VariantsVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoCanonical,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};

mod compress;
mod compute;

impl_encoding!("fastlanes.bitpacked", ids::FL_BITPACKED, BitPacked);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitPackedMetadata {
    validity: ValidityMetadata,
    bit_width: u8,
    offset: u16, // must be <1024
    patches: Option<PatchesMetadata>,
}

impl Display for BitPackedMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

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
            ((length + offset as usize + 1023) / 1024) * (128 * bit_width as usize);
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

        let metadata = BitPackedMetadata {
            validity: validity.to_metadata(length)?,
            offset,
            bit_width,
            patches: patches
                .as_ref()
                .map(|p| p.to_metadata(length, &dtype))
                .transpose()?,
        };

        let mut children = Vec::with_capacity(3);
        if let Some(p) = patches.as_ref() {
            children.push(p.indices().clone());
            children.push(p.values().clone());
        }
        if let Some(a) = validity.into_array() {
            children.push(a)
        }

        ArrayData::try_new_owned(
            &BitPackedEncoding,
            dtype,
            length,
            Arc::new(metadata),
            Some(packed),
            children.into(),
            StatsSet::default(),
        )?
        .try_into()
    }

    #[inline]
    pub fn packed(&self) -> &ByteBuffer {
        self.as_ref()
            .byte_buffer()
            .vortex_expect("BitPackedArray must contain packed buffer")
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

    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.metadata().bit_width
    }

    /// Access the patches array.
    ///
    /// If present, patches MUST be a `SparseArray` with equal-length to this array, and whose
    /// indices indicate the locations of patches. The indices must have non-zero length.
    #[inline]
    pub fn patches(&self) -> Option<Patches> {
        self.metadata().patches.as_ref().map(|patches| {
            Patches::new(
                self.len(),
                self.as_ref()
                    .child(0, &patches.indices_dtype(), patches.len())
                    .vortex_expect("BitPackedArray: patch indices"),
                self.as_ref()
                    .child(1, self.dtype(), patches.len())
                    .vortex_expect("BitPackedArray: patch values"),
            )
        })
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.metadata().offset
    }

    pub fn validity(&self) -> Validity {
        let validity_child_idx = if self.metadata().patches.is_some() {
            2
        } else {
            0
        };
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(validity_child_idx, &Validity::DTYPE, self.len())
                .vortex_expect("BitPackedArray: validity child")
        })
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
    pub fn encode(array: &ArrayData, bit_width: u8) -> VortexResult<Self> {
        if let Ok(parray) = PrimitiveArray::try_from(array.clone()) {
            bitpack_encode(parray, bit_width)
        } else {
            vortex_bail!("Bitpacking can only encode primitive arrays");
        }
    }

    /// Calculate the maximum value that **can** be contained by this array, given its bit-width.
    ///
    /// Note that this value need not actually be present in the array.
    #[inline]
    fn max_packed_value(&self) -> usize {
        (1 << self.bit_width()) - 1
    }
}

impl IntoCanonical for BitPackedArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        unpack(self).map(Canonical::Primitive)
    }
}

impl ValidityVTable<BitPackedArray> for BitPackedEncoding {
    fn is_valid(&self, array: &BitPackedArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &BitPackedArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<BitPackedArray> for BitPackedEncoding {
    fn accept(&self, array: &BitPackedArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.packed())?;
        if let Some(patches) = array.patches().as_ref() {
            visitor.visit_patches(patches)?;
        }
        visitor.visit_validity(&array.validity())
    }
}

impl StatisticsVTable<BitPackedArray> for BitPackedEncoding {}

impl ArrayTrait for BitPackedArray {}

impl VariantsVTable<BitPackedArray> for BitPackedEncoding {
    fn as_primitive_array<'a>(
        &self,
        array: &'a BitPackedArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for BitPackedArray {}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::validity::ValidityMetadata;
    use vortex_array::{IntoArrayData, IntoArrayVariant, IntoCanonical};
    use vortex_buffer::Buffer;
    use vortex_dtype::PType;

    use crate::{BitPackedArray, BitPackedMetadata};

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_bitpacked_metadata() {
        check_metadata(
            "bitpacked.metadata",
            BitPackedMetadata {
                patches: Some(PatchesMetadata::new(usize::MAX, PType::U64)),
                validity: ValidityMetadata::AllValid,
                offset: u16::MAX,
                bit_width: u8::MAX,
            },
        );
    }

    #[test]
    fn test_encode() {
        let values = [Some(1), None, Some(1), None, Some(1), None, Some(u64::MAX)];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let packed = BitPackedArray::encode(uncompressed.as_ref(), 1).unwrap();
        let expected = &[1, 0, 1, 0, 1, 0, u64::MAX];
        let results = packed
            .into_array()
            .into_primitive()
            .unwrap()
            .as_slice::<u64>()
            .to_vec();
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
                .into_canonical()
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            values.as_slice()
        );
    }
}
