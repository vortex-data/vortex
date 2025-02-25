use std::fmt::Debug;
use std::sync::{Arc, RwLock};

pub use compute::compute_min_max;
use num_traits::PrimInt;
pub use stats::compute_varbin_statistics;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{
    VortexExpect as _, VortexResult, VortexUnwrap as _, vortex_bail, vortex_err, vortex_panic,
};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::ArrayValidityImpl;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::arrays::varbin::serde::VarBinMetadata;
use crate::compute::scalar_at;
use crate::encoding::encoding_ids;
use crate::stats::StatsSet;
use crate::validity::Validity;
use crate::vtable::VTableRef;
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, Encoding, EncodingId, RkyvMetadata,
    try_from_array_ref,
};

mod accessor;
pub mod builder;
mod canonical;
mod compute;
mod serde;
mod stats;
mod variants;

#[derive(Clone, Debug)]
pub struct VarBinArray {
    dtype: DType,
    bytes: ByteBuffer,
    offsets: ArrayRef,
    validity: Validity,
    stats_set: Arc<RwLock<StatsSet>>,
}

try_from_array_ref!(VarBinArray);

pub struct VarBinEncoding;
impl Encoding for VarBinEncoding {
    const ID: EncodingId = EncodingId::new("vortex.varbin", encoding_ids::VAR_BIN);
    type Array = VarBinArray;
    type Metadata = RkyvMetadata<VarBinMetadata>;
}

impl VarBinArray {
    pub fn try_new(
        offsets: ArrayRef,
        bytes: ByteBuffer,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non nullable int", offsets.dtype());
        }
        if !matches!(dtype, DType::Binary(_) | DType::Utf8(_)) {
            vortex_bail!(MismatchedTypes: "utf8 or binary", dtype);
        }
        if dtype.is_nullable() == (validity == Validity::NonNullable) {
            vortex_bail!("incorrect validity {:?}", validity);
        }

        Ok(Self {
            dtype,
            bytes,
            offsets,
            validity,
            stats_set: Default::default(),
        })
    }

    #[inline]
    pub fn offsets(&self) -> &ArrayRef {
        &self.offsets
    }

    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Access the value bytes child buffer
    ///
    /// # Note
    ///
    /// Bytes child buffer is never sliced when the array is sliced so this can include values
    /// that are not logically present in the array. Users should prefer [sliced_bytes][Self::sliced_bytes]
    /// unless they're resolving values via the offset child array.
    #[inline]
    pub fn bytes(&self) -> &ByteBuffer {
        &self.bytes
    }

    /// Access value bytes child array limited to values that are logically present in
    /// the array unlike [bytes][Self::bytes].
    pub fn sliced_bytes(&self) -> ByteBuffer {
        let first_offset: usize = self.offset_at(0).vortex_expect("1st offset");
        let last_offset = self.offset_at(self.len()).vortex_expect("Last offset");

        self.bytes().slice(first_offset..last_offset)
    }

    pub fn from_vec<T: AsRef<[u8]>>(vec: Vec<T>, dtype: DType) -> Self {
        let size: usize = vec.iter().map(|v| v.as_ref().len()).sum();
        if size < u32::MAX as usize {
            Self::from_vec_sized::<u32, T>(vec, dtype)
        } else {
            Self::from_vec_sized::<u64, T>(vec, dtype)
        }
    }

    fn from_vec_sized<O, T>(vec: Vec<T>, dtype: DType) -> Self
    where
        O: NativePType + PrimInt,
        T: AsRef<[u8]>,
    {
        let mut builder = VarBinBuilder::<O>::with_capacity(vec.len());
        for v in vec {
            builder.append_value(v.as_ref());
        }
        builder.finish(dtype)
    }

    #[allow(clippy::same_name_method)]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinBuilder::<u32>::with_capacity(iter.size_hint().0);
        for v in iter {
            builder.append(v.as_ref().map(|o| o.as_ref()));
        }
        builder.finish(dtype)
    }

    pub fn from_iter_nonnull<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinBuilder::<u32>::with_capacity(iter.size_hint().0);
        for v in iter {
            builder.append_value(v);
        }
        builder.finish(dtype)
    }

    /// Get value offset at a given index
    ///
    /// Note: There's 1 more offsets than the elements in the array, thus last offset is at array length index
    pub fn offset_at(&self, index: usize) -> VortexResult<usize> {
        if index > self.len() + 1 {
            vortex_bail!(OutOfBounds: index, 0, self.len() + 1)
        }

        // TODO(ngates): PrimitiveArrayTrait should have get_scalar(idx) -> Option<T> method
        Ok(scalar_at(self.offsets(), index)
            .unwrap_or_else(|err| vortex_panic!(err, "Failed to get offset at index: {}", index))
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert offset to usize"))
    }

    /// Access value bytes at a given index
    ///
    /// Will return buffer referncing underlying data without performing a copy
    pub fn bytes_at(&self, index: usize) -> VortexResult<ByteBuffer> {
        let start = self.offset_at(index)?;
        let end = self.offset_at(index + 1)?;

        Ok(self.bytes().slice(start..end))
    }

    /// Consumes self, returning a tuple containing the `DType`, the `bytes` array,
    /// the `offsets` array, and the `validity`.
    pub fn into_parts(self) -> (DType, ByteBuffer, ArrayRef, Validity) {
        (self.dtype, self.bytes, self.offsets, self.validity)
    }
}

impl ArrayValidityImpl for VarBinArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len())
    }
}

impl ArrayImpl for VarBinArray {
    type Encoding = VarBinEncoding;

    fn _len(&self) -> usize {
        self.offsets().len().saturating_sub(1)
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&VarBinEncoding)
    }
}

impl ArrayStatisticsImpl for VarBinArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl From<Vec<&[u8]>> for VarBinArray {
    fn from(value: Vec<&[u8]>) -> Self {
        Self::from_vec(value, DType::Binary(Nullability::NonNullable))
    }
}

impl From<Vec<Vec<u8>>> for VarBinArray {
    fn from(value: Vec<Vec<u8>>) -> Self {
        Self::from_vec(value, DType::Binary(Nullability::NonNullable))
    }
}

impl From<Vec<String>> for VarBinArray {
    fn from(value: Vec<String>) -> Self {
        Self::from_vec(value, DType::Utf8(Nullability::NonNullable))
    }
}

impl From<Vec<&str>> for VarBinArray {
    fn from(value: Vec<&str>) -> Self {
        Self::from_vec(value, DType::Utf8(Nullability::NonNullable))
    }
}

impl<'a> FromIterator<Option<&'a [u8]>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a [u8]>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<Vec<u8>>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<Vec<u8>>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<String>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Utf8(Nullability::Nullable))
    }
}

impl<'a> FromIterator<Option<&'a str>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a str>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Utf8(Nullability::Nullable))
    }
}

pub fn varbin_scalar(value: ByteBuffer, dtype: &DType) -> Scalar {
    if matches!(dtype, DType::Utf8(_)) {
        Scalar::try_utf8(value, dtype.nullability())
            .map_err(|err| vortex_err!("Failed to create scalar from utf8 buffer: {}", err))
            .vortex_unwrap()
    } else {
        Scalar::binary(value, dtype.nullability())
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability};

    use crate::ArrayRef;
    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::arrays::varbin::VarBinArray;
    use crate::compute::{scalar_at, slice};
    use crate::validity::Validity;

    #[fixture]
    fn binary_array() -> ArrayRef {
        let values = Buffer::copy_from("hello worldhello world this is a long string".as_bytes());
        let offsets = PrimitiveArray::from_iter([0, 11, 44]);

        VarBinArray::try_new(
            offsets.into_array(),
            values,
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    pub fn test_scalar_at(binary_array: ArrayRef) {
        assert_eq!(binary_array.len(), 2);
        assert_eq!(scalar_at(&binary_array, 0).unwrap(), "hello world".into());
        assert_eq!(
            scalar_at(&binary_array, 1).unwrap(),
            "hello world this is a long string".into()
        )
    }

    #[rstest]
    pub fn slice_array(binary_array: ArrayRef) {
        let binary_arr = slice(&binary_array, 1, 2).unwrap();
        assert_eq!(
            scalar_at(&binary_arr, 0).unwrap(),
            "hello world this is a long string".into()
        );
    }
}
