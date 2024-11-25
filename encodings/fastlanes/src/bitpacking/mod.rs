use std::fmt::{Debug, Display};
use std::sync::Arc;

use ::serde::{Deserialize, Serialize};
pub use compress::*;
use fastlanes::BitPacking;
use vortex_array::array::{PrimitiveArray, SparseArray};
use vortex_array::encoding::ids;
use vortex_array::stats::{StatisticsVTable, StatsSet};
use vortex_array::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoCanonical,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};

mod compress;
mod compute;

impl_encoding!("fastlanes.bitpacked", ids::FL_BITPACKED, BitPacked);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitPackedMetadata {
    validity: ValidityMetadata,
    bit_width: u8,
    offset: u16, // must be <1024
    has_patches: bool,
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
    pub fn try_new(
        packed: Buffer,
        ptype: PType,
        validity: Validity,
        patches: Option<ArrayData>,
        bit_width: u8,
        len: usize,
    ) -> VortexResult<Self> {
        Self::try_new_from_offset(packed, ptype, validity, patches, bit_width, len, 0)
    }

    pub(crate) fn try_new_from_offset(
        packed: Buffer,
        ptype: PType,
        validity: Validity,
        patches: Option<ArrayData>,
        bit_width: u8,
        length: usize,
        offset: u16,
    ) -> VortexResult<Self> {
        let dtype = DType::Primitive(ptype, validity.nullability());

        if !dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "uint", &dtype);
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

        if let Some(parray) = patches.as_ref() {
            if parray.len() != length {
                vortex_bail!(
                    "Mismatched length in BitPackedArray between encoded {} and it's patches({}) {}",
                    length,
                    parray.encoding().id(),
                    parray.len()
                )
            }

            if SparseArray::try_from(parray.clone())?.indices().is_empty() {
                vortex_bail!("cannot construct BitPackedArray using patches without indices");
            }
        }

        let metadata = BitPackedMetadata {
            validity: validity.to_metadata(length)?,
            offset,
            bit_width,
            has_patches: patches.is_some(),
        };

        let mut children = Vec::with_capacity(2);
        if let Some(p) = patches {
            children.push(p);
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
    pub fn packed(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("BitPackedArray must contain packed buffer")
    }

    /// Access the slice of packed values as an array of `T`
    #[inline]
    pub fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: maybe_null_slice points to buffer memory that outlives the lifetime of `self`.
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
    pub fn patches(&self) -> Option<ArrayData> {
        self.metadata().has_patches.then(|| {
            self.as_ref()
                .child(
                    0,
                    &self.dtype().with_nullability(Nullability::Nullable),
                    self.len(),
                )
                .vortex_expect("BitPackedArray: patches child")
        })
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.metadata().offset
    }

    pub fn validity(&self) -> Validity {
        let validity_child_idx = if self.metadata().has_patches { 1 } else { 0 };

        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(validity_child_idx, &Validity::DTYPE, self.len())
                .vortex_expect("BitPackedArray: validity child")
        })
    }

    pub fn encode(array: &ArrayData, bit_width: u8) -> VortexResult<Self> {
        if let Ok(parray) = PrimitiveArray::try_from(array.clone()) {
            bitpack_encode(parray, bit_width)
        } else {
            vortex_bail!("Bitpacking can only encode primitive arrays");
        }
    }

    #[inline]
    pub fn max_packed_value(&self) -> usize {
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
            visitor.visit_child("patches", patches)?;
        }
        visitor.visit_validity(&array.validity())
    }
}

impl StatisticsVTable<BitPackedArray> for BitPackedEncoding {}

impl ArrayTrait for BitPackedArray {}

impl ArrayVariants for BitPackedArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for BitPackedArray {}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::{IntoArrayData, IntoArrayVariant};

    use crate::BitPackedArray;

    #[test]
    fn test_encode() {
        let values = vec![Some(1), None, Some(1), None, Some(1), None, Some(u64::MAX)];
        let uncompressed = PrimitiveArray::from_nullable_vec(values);
        let packed = BitPackedArray::encode(uncompressed.as_ref(), 1).unwrap();
        let expected = &[1, 0, 1, 0, 1, 0, u64::MAX];
        let results = packed
            .into_array()
            .into_primitive()
            .unwrap()
            .maybe_null_slice::<u64>()
            .to_vec();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_encode_too_wide() {
        let values = vec![Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_nullable_vec(values);
        let _packed = BitPackedArray::encode(uncompressed.as_ref(), 8)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedArray::encode(uncompressed.as_ref(), 9)
            .expect_err("Cannot pack value into larger width");
    }
}
