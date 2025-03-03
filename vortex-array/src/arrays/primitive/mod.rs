use std::fmt::Debug;
use std::iter;

mod accessor;

use arrow_buffer::BooleanBufferBuilder;
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{DType, NativePType, Nullability, PType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayVariantsImpl, Canonical, EmptyMetadata,
    Encoding, EncodingId, IntoArray, try_from_array_ref,
};

mod compute;
mod patch;
mod serde;
mod stats;

#[derive(Clone, Debug)]
pub struct PrimitiveArray {
    dtype: DType,
    buffer: ByteBuffer,
    validity: Validity,
    stats_set: ArrayStats,
}

try_from_array_ref!(PrimitiveArray);

pub struct PrimitiveEncoding;
impl Encoding for PrimitiveEncoding {
    type Array = PrimitiveArray;
    type Metadata = EmptyMetadata;
}

impl EncodingVTable for PrimitiveEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.primitive")
    }
}

impl PrimitiveArray {
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        let buffer = buffer.into();
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
        Self::new(Buffer::<T>::empty(), nullability.into())
    }

    pub fn from_byte_buffer(buffer: ByteBuffer, ptype: PType, validity: Validity) -> Self {
        match_each_native_ptype!(ptype, |$T| {
            Self::new::<$T>(Buffer::from_byte_buffer(buffer), validity)
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
        Self::new(values.freeze(), Validity::from(validity.finish()))
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
            .map_err(|buffer| PrimitiveArray::new(buffer, validity))
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
        PrimitiveArray::new(buffer.freeze(), validity)
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
                let val = val.to_canonical()?.into_bool()?;
                BufferMut::<R>::from_iter(buf_iter.zip(val.boolean_buffer()).map(f))
            }
        };
        Ok(PrimitiveArray::new(buffer.freeze(), validity.clone()))
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
        VTableRef::new_ref(&PrimitiveEncoding)
    }
}

impl ArrayStatisticsImpl for PrimitiveArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayVariantsImpl for PrimitiveArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for PrimitiveArray {}

impl<T: NativePType> FromIterator<T> for PrimitiveArray {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values = BufferMut::from_iter(iter);
        PrimitiveArray::new(values.freeze(), Validity::NonNullable)
    }
}

impl<T: NativePType> IntoArray for Buffer<T> {
    fn into_array(self) -> ArrayRef {
        PrimitiveArray::new(self, Validity::NonNullable).into_array()
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
        builder.extend_from_array(self)
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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::array::Array;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::test_harness::test_mask;
    use crate::validity::Validity;

    #[test]
    fn test_mask_primitive_array() {
        test_mask(&PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::NonNullable,
        ));
        test_mask(&PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::AllValid,
        ));
        test_mask(&PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::AllInvalid,
        ));
        test_mask(&PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        ));
    }
}
