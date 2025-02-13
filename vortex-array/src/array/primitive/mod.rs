use std::fmt::{Debug, Display};
use std::{iter, ptr};
mod accessor;

use arrow_buffer::BooleanBufferBuilder;
use serde::{Deserialize, Serialize};
use vortex_buffer::{Alignment, Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::builders::ArrayBuilder;
use crate::encoding::encoding_ids;
use crate::iter::Accessor;
use crate::stats::StatsSet;
use crate::validity::{Validity, ValidityMetadata};
use crate::variants::PrimitiveArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    CanonicalVTable, ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use crate::{impl_encoding, Array, Canonical, IntoArray, IntoCanonical, RkyvMetadata};

mod compute;
mod patch;
mod stats;

impl_encoding!(
    "vortex.primitive",
    encoding_ids::PRIMITIVE,
    Primitive,
    RkyvMetadata<PrimitiveMetadata>
);

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
    pub fn new<T: NativePType>(buffer: impl Into<Buffer<T>>, validity: Validity) -> Self {
        let buffer = buffer.into();
        let len = buffer.len();

        Self::try_from_parts(
            DType::from(T::PTYPE).with_nullability(validity.nullability()),
            len,
            RkyvMetadata(PrimitiveMetadata {
                validity: validity.to_metadata(len).vortex_expect("Invalid validity"),
            }),
            Some([buffer.into_byte_buffer()].into()),
            validity.into_array().map(|v| [v].into()),
            StatsSet::default(),
        )
        .vortex_expect("Should not fail to create PrimitiveArray")
    }

    pub fn empty<T: NativePType>(nullability: Nullability) -> Self {
        Self::new(
            Buffer::<T>::empty(),
            match nullability {
                Nullability::NonNullable => Validity::NonNullable,
                Nullability::Nullable => Validity::AllValid,
            },
        )
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

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("PrimitiveArray: validity child")
        })
    }

    pub fn byte_buffer(&self) -> &ByteBuffer {
        self.as_ref()
            .byte_buffer(0)
            .vortex_expect("Missing buffer in PrimitiveArray")
    }

    pub fn into_byte_buffer(self) -> ByteBuffer {
        self.into_array()
            .into_byte_buffer(0)
            .vortex_expect("PrimitiveArray must have a buffer")
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
        Buffer::from_byte_buffer(
            self.into_array()
                .into_byte_buffer(0)
                .vortex_expect("PrimitiveArray must have a buffer"),
        )
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
        let validity = self.validity();
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
        let validity = self.validity();
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
                let val_bb = val.clone().into_canonical()?.into_bool()?.boolean_buffer();
                BufferMut::<R>::from_iter(buf_iter.zip(&val_bb).map(f))
            }
        };
        Ok(PrimitiveArray::new(buffer.freeze(), validity))
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

        PrimitiveArray::from_byte_buffer(self.byte_buffer().clone(), ptype, self.validity())
    }
}

impl ValidateVTable<PrimitiveArray> for PrimitiveEncoding {
    fn validate(&self, array: &PrimitiveArray) -> VortexResult<()> {
        if array.as_ref().nbuffers() != 1 {
            vortex_bail!(
                "PrimitiveArray: expected 1 buffer, found {}",
                array.as_ref().nbuffers()
            );
        }

        match_each_native_ptype!(array.ptype(), |$T| {
            if !array
                .byte_buffer()
                .alignment()
                .is_aligned_to(Alignment::of::<$T>())
            {
                vortex_bail!("PrimitiveArray: buffer is not aligned to {}", stringify!($T));
            }
        });

        Ok(())
    }
}

impl VariantsVTable<PrimitiveArray> for PrimitiveEncoding {
    fn as_primitive_array<'a>(
        &self,
        array: &'a PrimitiveArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
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
        PrimitiveArray::new(values.freeze(), Validity::NonNullable)
    }
}

impl<T: NativePType> IntoArray for Buffer<T> {
    fn into_array(self) -> Array {
        PrimitiveArray::new(self, Validity::NonNullable).into_array()
    }
}

impl<T: NativePType> IntoArray for BufferMut<T> {
    fn into_array(self) -> Array {
        self.freeze().into_array()
    }
}

impl CanonicalVTable<PrimitiveArray> for PrimitiveEncoding {
    fn into_canonical(&self, array: PrimitiveArray) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(array))
    }

    fn canonicalize_into(
        &self,
        array: PrimitiveArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(array.into_array())
    }
}

impl ValidityVTable<PrimitiveArray> for PrimitiveEncoding {
    fn is_valid(&self, array: &PrimitiveArray, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn all_valid(&self, array: &PrimitiveArray) -> VortexResult<bool> {
        array.validity().all_valid()
    }

    fn all_invalid(&self, array: &PrimitiveArray) -> VortexResult<bool> {
        array.validity().all_invalid()
    }

    fn validity_mask(&self, array: &PrimitiveArray) -> VortexResult<Mask> {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<PrimitiveArray> for PrimitiveEncoding {
    fn accept(&self, array: &PrimitiveArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.byte_buffer())?;
        visitor.visit_validity(&array.validity())
    }
}
