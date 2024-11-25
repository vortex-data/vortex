use std::fmt::{Debug, Display};
use std::ptr;
use std::sync::Arc;
mod accessor;

use arrow_buffer::{ArrowNativeType, Buffer as ArrowBuffer, MutableBuffer};
use bytes::Bytes;
use itertools::Itertools;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::iter::Accessor;
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{ArrayVariants, PrimitiveArrayTrait};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoCanonical,
};

mod compute;
mod stats;

impl_encoding!("vortex.primitive", ids::PRIMITIVE, Primitive);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrimitiveMetadata {
    validity: ValidityMetadata,
}

impl Display for PrimitiveMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl PrimitiveArray {
    pub fn new(buffer: Buffer, ptype: PType, validity: Validity) -> Self {
        let length = match_each_native_ptype!(ptype, |$P| {
            let (prefix, values, suffix) = unsafe { buffer.align_to::<$P>() };
            assert!(
                prefix.is_empty() && suffix.is_empty() && (buffer.as_ptr() as usize) % std::mem::size_of::<$P>() == 0,
                "buffer is not aligned: {:?}",
                buffer.as_ptr()
            );
            values.len()
        });

        ArrayData::try_new_owned(
            &PrimitiveEncoding,
            DType::from(ptype).with_nullability(validity.nullability()),
            length,
            Arc::new(PrimitiveMetadata {
                validity: validity
                    .to_metadata(length)
                    .vortex_expect("Invalid validity"),
            }),
            Some(buffer),
            validity.into_array().into_iter().collect_vec().into(),
            StatsSet::default(),
        )
        .and_then(|data| data.try_into())
        .vortex_expect("Should not fail to create PrimitiveArray")
    }

    pub fn from_vec<T: NativePType>(values: Vec<T>, validity: Validity) -> Self {
        match_each_native_ptype!(T::PTYPE, |$P| {
            PrimitiveArray::new(
                ArrowBuffer::from(MutableBuffer::from(unsafe { std::mem::transmute::<Vec<T>, Vec<$P>>(values) })).into(),
                T::PTYPE,
                validity,
            )
        })
    }

    pub fn from_nullable_vec<T: NativePType>(values: Vec<Option<T>>) -> Self {
        let elems: Vec<T> = values.iter().map(|v| v.unwrap_or_default()).collect();
        let validity = Validity::from_iter(values.iter().map(|v| v.is_some()));
        Self::from_vec(elems, validity)
    }

    /// Creates a new array of type U8
    pub fn from_bytes(bytes: Bytes, validity: Validity) -> Self {
        let buffer = Buffer::from(bytes);

        PrimitiveArray::new(buffer, PType::U8, validity)
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("PrimitiveArray: validity child")
        })
    }

    pub fn buffer(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("Missing buffer in PrimitiveArray")
    }

    pub fn maybe_null_slice<T: NativePType>(&self) -> &[T] {
        assert_eq!(
            T::PTYPE,
            self.ptype(),
            "Attempted to get slice of type {} from array of type {}",
            T::PTYPE,
            self.ptype(),
        );

        let raw_slice = self.buffer().as_slice();
        let typed_len = raw_slice.len() / size_of::<T>();
        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr().cast(), typed_len) }
    }

    /// Convert the array into a mutable vec of the given type.
    /// If possible, this will be zero-copy.
    pub fn into_maybe_null_slice<T: NativePType + ArrowNativeType>(self) -> Vec<T> {
        assert_eq!(
            T::PTYPE,
            self.ptype(),
            "Attempted to get maybe_null_slice of type {} from array of type {}",
            T::PTYPE,
            self.ptype(),
        );
        self.into_buffer().into_vec::<T>().unwrap_or_else(|b| {
            let (prefix, values, suffix) = unsafe { b.as_ref().align_to::<T>() };
            assert!(prefix.is_empty() && suffix.is_empty());
            Vec::from(values)
        })
    }

    pub fn get_as_cast<T: NativePType>(&self, idx: usize) -> T {
        match_each_native_ptype!(self.ptype(), |$P| {
            T::from(self.maybe_null_slice::<$P>()[idx]).expect("failed to cast")
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

        PrimitiveArray::new(self.buffer().clone(), ptype, self.validity())
    }

    pub fn patch<P: AsPrimitive<usize>, T: NativePType + ArrowNativeType>(
        self,
        positions: &[P],
        values: &[T],
        values_validity: Validity,
    ) -> VortexResult<Self> {
        if positions.len() != values.len() {
            vortex_bail!(
                "Positions and values passed to patch had different lengths {} and {}",
                positions.len(),
                values.len()
            );
        }
        if let Some(last_pos) = positions.last() {
            if last_pos.as_() >= self.len() {
                vortex_bail!(OutOfBounds: last_pos.as_(), 0, self.len())
            }
        }

        if self.ptype() != T::PTYPE {
            vortex_bail!(MismatchedTypes: self.dtype(), T::PTYPE)
        }

        let result_validity = self
            .validity()
            .patch(self.len(), positions, values_validity)?;
        let mut own_values = self.into_maybe_null_slice::<T>();
        for (idx, value) in positions.iter().zip_eq(values) {
            own_values[idx.as_()] = *value;
        }

        Ok(Self::from_vec(own_values, result_validity))
    }

    pub fn into_buffer(self) -> Buffer {
        self.into_array()
            .into_buffer()
            .vortex_expect("PrimitiveArray must have a buffer")
    }
}

impl ArrayTrait for PrimitiveArray {}

impl ArrayVariants for PrimitiveArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl<T: NativePType> Accessor<T> for PrimitiveArray {
    fn array_len(&self) -> usize {
        self.len()
    }

    fn is_valid(&self, index: usize) -> bool {
        ArrayValidity::is_valid(self, index)
    }

    #[inline]
    fn value_unchecked(&self, index: usize) -> T {
        self.maybe_null_slice::<T>()[index]
    }

    fn array_validity(&self) -> Validity {
        self.validity()
    }

    #[inline]
    fn decode_batch(&self, start_idx: usize) -> Vec<T> {
        let batch_size = <Self as Accessor<T>>::batch_size(self, start_idx);
        let mut v = Vec::<T>::with_capacity(batch_size);
        let null_slice = self.maybe_null_slice::<T>();

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

impl<T: NativePType> From<Vec<T>> for PrimitiveArray {
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values, Validity::NonNullable)
    }
}

impl<T: NativePType> IntoArrayData for Vec<T> {
    fn into_array(self) -> ArrayData {
        PrimitiveArray::from(self).into_array()
    }
}

impl IntoCanonical for PrimitiveArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(self))
    }
}

impl ValidityVTable<PrimitiveArray> for PrimitiveEncoding {
    fn is_valid(&self, array: &PrimitiveArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &PrimitiveArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<PrimitiveArray> for PrimitiveEncoding {
    fn accept(&self, array: &PrimitiveArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.buffer())?;
        visitor.visit_validity(&array.validity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::slice;
    use crate::IntoArrayVariant;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::from_vec(vec![2u32; 10], Validity::AllValid);
        let sliced = slice(input, 2, 8).unwrap();
        assert_eq!(
            sliced
                .into_primitive()
                .unwrap()
                .into_maybe_null_slice::<u32>(),
            vec![2u32; 6]
        );
    }
}
