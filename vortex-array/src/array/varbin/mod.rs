use std::fmt::{Debug, Display};

use num_traits::{AsPrimitive, PrimInt};
use rkyv::from_bytes;
use serde::{Deserialize, Serialize};
pub use stats::compute_varbin_statistics;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexError, VortexExpect as _, VortexResult,
    VortexUnwrap as _,
};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::varbin::builder::VarBinBuilder;
use crate::array::{StructMetadata, VarBinViewArray, VarBinViewMetadata};
use crate::compute::scalar_at;
use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validate::ValidateVTable;
use crate::validity::{Validity, ValidityMetadata};
use crate::variants::PrimitiveArrayTrait;
use crate::{impl_encoding, ArrayDType, ArrayData, ArrayLen, DeserializeMetadata, RkyvMetadata};

mod accessor;
mod array;
mod arrow;
pub mod builder;
mod canonical;
mod compute;
mod stats;
mod variants;

impl_encoding!(
    "vortex.varbin",
    ids::VAR_BIN,
    VarBin,
    RkyvMetadata<VarBinMetadata>
);

#[derive(
    Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct VarBinMetadata {
    pub(crate) validity: ValidityMetadata,
    pub(crate) offsets_ptype: PType,
    pub(crate) bytes_len: usize,
}

impl Display for VarBinMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl VarBinArray {
    pub fn try_new(
        offsets: ArrayData,
        bytes: ByteBuffer,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non nullable int", offsets.dtype());
        }
        let offsets_ptype = PType::try_from(offsets.dtype()).vortex_unwrap();

        if !matches!(dtype, DType::Binary(_) | DType::Utf8(_)) {
            vortex_bail!(MismatchedTypes: "utf8 or binary", dtype);
        }
        if dtype.is_nullable() == (validity == Validity::NonNullable) {
            vortex_bail!("incorrect validity {:?}", validity);
        }

        let length = offsets.len() - 1;

        let metadata = VarBinMetadata {
            validity: validity.to_metadata(offsets.len() - 1)?,
            offsets_ptype,
            bytes_len: bytes.len(),
        };

        let children = match validity.into_array() {
            Some(validity) => {
                vec![offsets, validity]
            }
            None => {
                vec![offsets]
            }
        };

        Self::try_from_parts(
            dtype,
            length,
            RkyvMetadata(metadata),
            Some([bytes].into()),
            Some(children.into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn offsets(&self) -> ArrayData {
        self.as_ref()
            .child(
                0,
                &DType::Primitive(self.metadata().offsets_ptype, Nullability::NonNullable),
                self.len() + 1,
            )
            .vortex_expect("Missing offsets in VarBinArray")
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(1, &Validity::DTYPE, self.len())
                .vortex_expect("VarBinArray: validity child")
        })
    }

    /// Access the value bytes child buffer
    ///
    /// # Note
    ///
    /// Bytes child buffer is never sliced when the array is sliced so this can include values
    /// that are not logically present in the array. Users should prefer [sliced_bytes][Self::sliced_bytes]
    /// unless they're resolving values via the offset child array.
    #[inline]
    pub fn bytes(&self) -> ByteBuffer {
        self.as_ref()
            .byte_buffer(0)
            .vortex_expect("Missing data buffer")
            .clone()
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
            builder.push_value(v.as_ref());
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
            builder.push(v.as_ref().map(|o| o.as_ref()));
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
            builder.push_value(v);
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

        Ok(PrimitiveArray::maybe_from(self.offsets())
            .map(|p| {
                match_each_native_ptype!(p.ptype(), |$P| {
                    p.as_slice::<$P>()[index].as_()
                })
            })
            .unwrap_or_else(|| {
                scalar_at(self.offsets(), index)
                    .unwrap_or_else(|err| {
                        vortex_panic!(err, "Failed to get offset at index: {}", index)
                    })
                    .as_ref()
                    .try_into()
                    .vortex_expect("Failed to convert offset to usize")
            }))
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
    pub fn into_parts(self) -> (DType, ByteBuffer, ArrayData, Validity) {
        (
            self.dtype().clone(),
            self.bytes(),
            self.offsets(),
            self.validity(),
        )
    }
}

impl ValidateVTable<VarBinArray> for VarBinEncoding {}

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

    use crate::array::primitive::PrimitiveArray;
    use crate::array::varbin::VarBinArray;
    use crate::compute::{scalar_at, slice};
    use crate::validity::Validity;
    use crate::{ArrayData, IntoArrayData};

    #[fixture]
    fn binary_array() -> ArrayData {
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
    pub fn test_scalar_at(binary_array: ArrayData) {
        assert_eq!(binary_array.len(), 2);
        assert_eq!(scalar_at(&binary_array, 0).unwrap(), "hello world".into());
        assert_eq!(
            scalar_at(&binary_array, 1).unwrap(),
            "hello world this is a long string".into()
        )
    }

    #[rstest]
    pub fn slice_array(binary_array: ArrayData) {
        let binary_arr = slice(&binary_array, 1, 2).unwrap();
        assert_eq!(
            scalar_at(&binary_arr, 0).unwrap(),
            "hello world this is a long string".into()
        );
    }
}
