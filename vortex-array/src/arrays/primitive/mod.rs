use std::fmt::{Debug, Display};
use std::sync::{Arc, RwLock};
use std::{iter, ptr};

mod accessor;

use arrow_buffer::BooleanBufferBuilder;
use serde::{Deserialize, Serialize};
use vortex_buffer::{Alignment, Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::arrays::ConstantEncoding;
use crate::builders::ArrayBuilder;
use crate::encoding::encoding_ids;
use crate::iter::Accessor;
use crate::stats::{Stat, StatsSet};
use crate::validity::{Validity, ValidityMetadata};
use crate::variants::PrimitiveArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::vtable::VTableRef;
use crate::{
    validity, Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayVariantsImpl, ArrayVisitorImpl,
    Canonical, EmptyMetadata, Encoding, EncodingId, IntoArray, RkyvMetadata,
};

mod compute;
mod patch;
mod stats;

#[derive(Clone, Debug)]
pub struct PrimitiveArray {
    dtype: DType,
    buffer: ByteBuffer,
    validity: Validity,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct PrimitiveEncoding;
impl Encoding for PrimitiveEncoding {
    const ID: EncodingId = EncodingId::new("vortex.primitive", encoding_ids::PRIMITIVE);
    type Array = PrimitiveArray;
    type Metadata = EmptyMetadata;
}

#[derive(
    Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[repr(C)]
pub struct PrimitiveMetadata {
    pub(crate) validity: ValidityMetadata,
}

impl Display for PrimitiveMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl PrimitiveArray {
    /// Create a new [`PrimitiveArray`] from an all-valid buffer with the given nullability.
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, nullability: Nullability) -> Self {
        let buffer = buffer.into().into_byte_buffer();
        Self {
            dtype: DType::Primitive(T::PTYPE, nullability),
            buffer,
            validity: nullability.into(),
            stats_set: Default::default(),
        }
    }

    pub fn new_with_validity<T: NativePType>(buffer: Buffer<T>, validity: Validity) -> Self {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() != len {
                vortex_panic!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len(),
                    len
                );
            }
        }
        Self {
            dtype: DType::Primitive(T::PTYPE, validity.nullability()),
            buffer: buffer.into_byte_buffer(),
            validity,
            stats_set: Default::default(),
        }
    }

    pub fn empty<T: NativePType>(nullability: Nullability) -> Self {
        Self::new(Buffer::<T>::empty(), nullability)
    }

    pub fn from_byte_buffer(buffer: ByteBuffer, ptype: PType, validity: Validity) -> Self {
        match_each_native_ptype!(ptype, |$T| {
            Self::new_with_validity::<$T>(Buffer::from_byte_buffer(buffer), validity)
        })
    }

    /// Create a PrimitiveArray from an iterator of `T`.
    /// NOTE: we cannot impl FromIterator trait since it conflicts with `FromIterator<T>`.
    pub fn from_option_iter<T: NativePType, I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut values = BufferMut::with_capacity(iter.size_hint().0);
        let mut validity = BooleanBufferBuilder::new(values.capacity());

        for i in iter {
            match i {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(e) => {
                    validity.append(true);
                    values.push(e);
                }
            }
        }
        Self::new_with_validity(values.freeze(), Validity::from(validity.finish()))
    }

    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    pub fn byte_buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    pub fn into_byte_buffer(self) -> ByteBuffer {
        self.buffer
    }

    pub fn buffer<T: NativePType>(&self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        Buffer::from_byte_buffer(self.byte_buffer().clone())
    }

    pub fn into_buffer<T: NativePType>(self) -> Buffer<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        Buffer::from_byte_buffer(self.buffer)
    }

    /// Extract a mutable buffer from the PrimitiveArray. Attempts to do this with zero-copy
    /// if the buffer is uniquely owned, otherwise will make a copy.
    pub fn into_buffer_mut<T: NativePType>(self) -> BufferMut<T> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer_mut of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        self.into_buffer()
            .try_into_mut()
            .unwrap_or_else(|buffer| BufferMut::<T>::copy_from(&buffer))
    }

    /// Try to extract a mutable buffer from the PrimitiveArray with zero copy.
    #[allow(clippy::panic_in_result_fn)]
    pub fn try_into_buffer_mut<T: NativePType>(self) -> Result<BufferMut<T>, PrimitiveArray> {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get buffer_mut of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        let validity = self.validity().clone();
        Buffer::<T>::from_byte_buffer(self.into_byte_buffer())
            .try_into_mut()
            .map_err(|buffer| PrimitiveArray::new_with_validity(buffer, validity))
    }

    /// Map each element in the array to a new value.
    ///
    /// This ignores validity and maps over all maybe-null elements.
    ///
    /// TODO(ngates): we could be smarter here if validity is sparse and only run the function
    ///   over the valid elements.
    pub fn map_each<T, R, F>(self, f: F) -> PrimitiveArray
    where
        T: NativePType,
        R: NativePType,
        F: FnMut(T) -> R,
    {
        let validity = self.validity().clone();
        let buffer = match self.try_into_buffer_mut() {
            Ok(buffer_mut) => buffer_mut.map_each(f),
            Err(parray) => BufferMut::<R>::from_iter(parray.buffer::<T>().iter().copied().map(f)),
        };
        PrimitiveArray::new_with_validity(buffer.freeze(), validity)
    }

    /// Map each element in the array to a new value.
    ///
    /// This doesn't ignore validity and maps over all maybe-null elements, with a bool true if
    /// valid and false otherwise.
    pub fn map_each_with_validity<T, R, F>(self, f: F) -> VortexResult<PrimitiveArray>
    where
        T: NativePType,
        R: NativePType,
        F: FnMut((T, bool)) -> R,
    {
        let validity = self.validity();

        let buf_iter = self.buffer::<T>().into_iter();

        let buffer = match &validity {
            Validity::NonNullable | Validity::AllValid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(true)).map(f))
            }
            Validity::AllInvalid => {
                BufferMut::<R>::from_iter(buf_iter.zip(iter::repeat(false)).map(f))
            }
            Validity::Array(val) => {
                let val_bb = val.to_canonical()?.into_bool()?.boolean_buffer();
                BufferMut::<R>::from_iter(buf_iter.zip(val_bb).map(f))
            }
        };
        Ok(PrimitiveArray::new_with_validity(
            buffer.freeze(),
            validity.clone(),
        ))
    }

    /// Return a slice of the array's buffer.
    ///
    /// NOTE: these values may be nonsense if the validity buffer indicates that the value is null.
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get slice of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }
        let length = self.len();
        let raw_slice = self.byte_buffer().as_slice();
        debug_assert_eq!(raw_slice.len() / size_of::<T>(), length);
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), length) }
    }

    pub fn get_as_cast<T: NativePType>(&self, idx: usize) -> T {
        match_each_native_ptype!(self.ptype(), |$P| {
            T::from(self.as_slice::<$P>()[idx]).expect("failed to cast")
        })
    }

    pub fn reinterpret_cast(&self, ptype: PType) -> Self {
        if self.ptype() == ptype {
            return self.clone();
        }

        assert_eq!(
            self.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        PrimitiveArray::from_byte_buffer(self.byte_buffer().clone(), ptype, self.validity().clone())
    }
}

impl ArrayImpl for PrimitiveArray {
    type Encoding = PrimitiveEncoding;

    fn _len(&self) -> usize {
        self.byte_buffer().len() / self.ptype().byte_width()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }
    fn _vtable(&self) -> VTableRef {
        VTableRef::from_static(&PrimitiveEncoding)
    }
}

impl ArrayStatisticsImpl for PrimitiveArray {
    fn stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl ArrayVariantsImpl for PrimitiveArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl<T: NativePType> Accessor<T> for PrimitiveArray {
    #[inline]
    fn value_unchecked(&self, index: usize) -> T {
        self.as_slice::<T>()[index]
    }

    #[inline]
    fn decode_batch(&self, start_idx: usize) -> Vec<T> {
        let batch_size = <Self as Accessor<T>>::batch_size(self, start_idx);
        let mut v = Vec::<T>::with_capacity(batch_size);
        let null_slice = self.as_slice::<T>();

        unsafe {
            v.set_len(batch_size);
            ptr::copy_nonoverlapping(
                null_slice.as_ptr().add(start_idx),
                v.as_mut_ptr(),
                batch_size,
            );
        }

        v
    }
}

impl PrimitiveArrayTrait for PrimitiveArray {}

impl<T: NativePType> FromIterator<T> for PrimitiveArray {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values = BufferMut::from_iter(iter);
        PrimitiveArray::new(values.freeze(), Nullability::NonNullable)
    }
}

impl<T: NativePType> IntoArray for Buffer<T> {
    fn into_array(self) -> ArrayRef {
        PrimitiveArray::new(self, Nullability::NonNullable).into_array()
    }
}

impl<T: NativePType> IntoArray for BufferMut<T> {
    fn into_array(self) -> ArrayRef {
        self.freeze().into_array()
    }
}

impl ArrayCanonicalImpl for PrimitiveArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(self.clone()))
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(&self.to_array())
    }
}

impl ArrayValidityImpl for PrimitiveArray {
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

impl ArrayVisitorImpl for PrimitiveArray {
    fn _accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(self.byte_buffer())?;
        visitor.visit_validity(self.validity())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;
    use crate::validity::Validity;
    #[test]
    fn test_mask_primitive_array() {
        test_mask(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array());
        test_mask(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array());
        test_mask(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllInvalid).into_array());
        test_mask(
            PrimitiveArray::new(
                buffer![0, 1, 2, 3, 4],
                Validity::Array(
                    BoolArray::from_iter([true, false, true, false, true]).into_array(),
                ),
            )
            .into_array(),
        );
    }
}
