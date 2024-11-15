use std::fmt::{Debug, Display};
use std::mem::{transmute, MaybeUninit};
use std::ptr;
use std::sync::Arc;

use arrow_buffer::{ArrowNativeType, Buffer as ArrowBuffer, MutableBuffer};
use bytes::Bytes;
use itertools::Itertools;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::elementwise::{dyn_cast_array_iter, BinaryFn, UnaryFn};
use crate::encoding::ids;
use crate::iter::{Accessor, AccessorRef, Batch, ITER_BATCH_SIZE};
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use crate::variants::{ArrayVariants, PrimitiveArrayTrait};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayTrait, Canonical, IntoArrayData, IntoCanonical,
    TypedArray,
};

mod accessor;
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

        Self {
            typed: TypedArray::try_from_parts(
                DType::from(ptype).with_nullability(validity.nullability()),
                length,
                PrimitiveMetadata {
                    validity: validity
                        .to_metadata(length)
                        .vortex_expect("Invalid validity"),
                },
                Some(buffer),
                validity.into_array().into_iter().collect_vec().into(),
                StatsSet::new(),
            )
            .vortex_expect("PrimitiveArray::new should never fail!"),
        }
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
        let validity = Validity::from(values.iter().map(|v| v.is_some()).collect::<Vec<_>>());
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

macro_rules! primitive_accessor_ref {
    ($self:expr, $ptype:ident) => {
        match $self.dtype() {
            DType::Primitive(PType::$ptype, _) => {
                let accessor = Arc::new($self.clone());
                Some(accessor)
            }
            _ => None,
        }
    };
}

impl PrimitiveArrayTrait for PrimitiveArray {
    fn f32_accessor(&self) -> Option<AccessorRef<f32>> {
        primitive_accessor_ref!(self, F32)
    }

    fn f64_accessor(&self) -> Option<AccessorRef<f64>> {
        primitive_accessor_ref!(self, F64)
    }

    fn u8_accessor(&self) -> Option<AccessorRef<u8>> {
        primitive_accessor_ref!(self, U8)
    }

    fn u16_accessor(&self) -> Option<AccessorRef<u16>> {
        primitive_accessor_ref!(self, U16)
    }

    fn u32_accessor(&self) -> Option<AccessorRef<u32>> {
        primitive_accessor_ref!(self, U32)
    }

    fn u64_accessor(&self) -> Option<AccessorRef<u64>> {
        primitive_accessor_ref!(self, U64)
    }

    fn i8_accessor(&self) -> Option<AccessorRef<i8>> {
        primitive_accessor_ref!(self, I8)
    }

    fn i16_accessor(&self) -> Option<AccessorRef<i16>> {
        primitive_accessor_ref!(self, I16)
    }

    fn i32_accessor(&self) -> Option<AccessorRef<i32>> {
        primitive_accessor_ref!(self, I32)
    }

    fn i64_accessor(&self) -> Option<AccessorRef<i64>> {
        primitive_accessor_ref!(self, I64)
    }
}

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

impl ArrayValidity for PrimitiveArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl AcceptArrayVisitor for PrimitiveArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(self.buffer())?;
        visitor.visit_validity(&self.validity())
    }
}

impl ArrayData {
    pub fn as_primitive(&self) -> PrimitiveArray {
        PrimitiveArray::try_from(self).vortex_expect("Expected primitive array")
    }
}

// This is an arbitrary value, tried a few seems like this is a better value than smaller ones,
// I assume there's some hardware dependency here but this seems to be good enough
const CHUNK_SIZE: usize = 1024;

impl UnaryFn for PrimitiveArray {
    fn unary<I: NativePType, O: NativePType, F: Fn(I) -> O>(
        &self,
        unary_fn: F,
    ) -> VortexResult<ArrayData> {
        let data = self.maybe_null_slice::<I>();
        let mut output: Vec<MaybeUninit<O>> = Vec::with_capacity(data.len());
        // Safety: we are going to apply the fn to every element and store it so the full length will be utilized
        unsafe { output.set_len(data.len()) };

        let chunks = data.chunks_exact(CHUNK_SIZE);

        // We start with the reminder because of ownership
        let reminder_start_idx = data.len() - (data.len() % CHUNK_SIZE);
        for (index, item) in chunks.remainder().iter().enumerate() {
            // Safety: This access is bound by the same range as the output's capacity and length, so its within the Vec's allocated memory
            unsafe {
                *output.get_unchecked_mut(reminder_start_idx + index) =
                    MaybeUninit::new(unary_fn(*item));
            }
        }

        let mut offset = 0;

        for chunk in chunks {
            // We know the size of the chunk, and we know output is the same length as the input array
            let chunk: [I; CHUNK_SIZE] = chunk.try_into()?;
            let output_slice: &mut [_; CHUNK_SIZE] =
                (&mut output[offset..offset + CHUNK_SIZE]).try_into()?;

            for idx in 0..CHUNK_SIZE {
                output_slice[idx] = MaybeUninit::new(unary_fn(chunk[idx]));
            }

            offset += CHUNK_SIZE;
        }

        // Safety: `MaybeUninit` is a transparent struct and we know the actual length of the vec.
        let output = unsafe { transmute::<Vec<MaybeUninit<O>>, Vec<O>>(output) };

        Ok(PrimitiveArray::from_vec(output, self.validity()).into_array())
    }
}

impl BinaryFn for PrimitiveArray {
    fn binary<I: NativePType, U: NativePType, O: NativePType, F: Fn(I, U) -> O>(
        &self,
        rhs: ArrayData,
        binary_fn: F,
    ) -> VortexResult<ArrayData> {
        if self.len() != rhs.len() {
            vortex_bail!(InvalidArgument: "Both arguments to `binary` should be of the same length");
        }
        if !self.dtype().eq_ignore_nullability(rhs.dtype()) {
            vortex_bail!(MismatchedTypes: self.dtype(), rhs.dtype());
        }

        if PType::try_from(self.dtype())? != I::PTYPE {
            vortex_bail!(MismatchedTypes: self.dtype(), I::PTYPE);
        }

        let lhs = self.maybe_null_slice::<I>();

        let mut output: Vec<MaybeUninit<O>> = Vec::with_capacity(self.len());
        // Safety: we are going to apply the fn to every element and store it so the full length will be utilized
        unsafe { output.set_len(self.len()) };

        let validity = self
            .validity()
            .and(rhs.with_dyn(|a| a.logical_validity().into_validity()))?;

        let mut idx_offset = 0;
        let rhs_iter = dyn_cast_array_iter::<U>(&rhs);

        for batch in rhs_iter {
            let batch_len = batch.len();
            process_batch(
                &lhs[idx_offset..idx_offset + batch_len],
                batch,
                &binary_fn,
                idx_offset,
                output.as_mut_slice(),
            );
            idx_offset += batch_len;
        }

        // Safety: `MaybeUninit` is a transparent struct and we know the actual length of the vec.
        let output = unsafe { transmute::<Vec<MaybeUninit<O>>, Vec<O>>(output) };

        Ok(PrimitiveArray::from_vec(output, validity).into_array())
    }
}

#[allow(clippy::unwrap_used)]
fn process_batch<I: NativePType, U: NativePType, O: NativePType, F: Fn(I, U) -> O>(
    lhs: &[I],
    batch: Batch<U>,
    f: F,
    idx_offset: usize,
    output: &mut [MaybeUninit<O>],
) {
    assert_eq!(batch.len(), lhs.len());

    if batch.len() == ITER_BATCH_SIZE {
        let lhs: [I; ITER_BATCH_SIZE] = lhs.try_into().unwrap();
        let rhs: [U; ITER_BATCH_SIZE] = batch.data().try_into().unwrap();
        // We know output is of the same length and lhs/rhs
        let output_slice: &mut [_; ITER_BATCH_SIZE] = (&mut output
            [idx_offset..idx_offset + ITER_BATCH_SIZE])
            .try_into()
            .unwrap();

        for idx in 0..ITER_BATCH_SIZE {
            unsafe {
                *output_slice.get_unchecked_mut(idx) = MaybeUninit::new(f(lhs[idx], rhs[idx]));
            }
        }
    } else {
        for (idx, rhs_item) in batch.data().iter().enumerate() {
            // Safety: output is the same length as the original array, so we know these are still valid indexes
            unsafe {
                *output.get_unchecked_mut(idx + idx_offset) =
                    MaybeUninit::new(f(lhs[idx], *rhs_item));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use super::*;
    use crate::compute::slice;
    use crate::IntoArrayVariant;

    #[test]
    fn batched_iter() {
        let v = PrimitiveArray::from_vec((0_u32..10_000).collect(), Validity::AllValid);
        let iter = v.u32_iter().unwrap();

        let mut items_counter = 0;

        for batch in iter {
            let batch_size = batch.len();
            for idx in 0..batch_size {
                assert!(batch.is_valid(idx));
                assert_eq!((items_counter + idx) as u32, unsafe {
                    *batch.get_unchecked(idx)
                });
            }

            items_counter += batch_size;
        }
    }

    #[test]
    fn flattened_iter() {
        let v = PrimitiveArray::from_vec((0_u32..10_000).collect(), Validity::AllValid);
        let iter = v.u32_iter().unwrap();

        for (idx, v) in iter.flatten().enumerate() {
            assert_eq!(idx as u32, v.unwrap());
        }
    }

    #[test]
    fn binary_fn_example() {
        let input = PrimitiveArray::from_vec(vec![2u32, 2, 2, 2], Validity::AllValid);

        let scalar = Scalar::from(2u32);

        let o = input
            .unary(move |v: u32| {
                let scalar_v = u32::try_from(&scalar).unwrap();
                if v == scalar_v {
                    1_u8
                } else {
                    0_u8
                }
            })
            .unwrap();

        let output_iter = o
            .with_dyn(|a| a.as_primitive_array_unchecked().u8_iter())
            .unwrap()
            .flatten();

        for v in output_iter {
            assert_eq!(v.unwrap(), 1);
        }
    }

    #[test]
    fn unary_fn_example() {
        let input = PrimitiveArray::from_vec(vec![2u32, 2, 2, 2], Validity::AllValid);
        let output = input.unary(|u: u32| u + 1).unwrap();

        for o in output
            .with_dyn(|a| a.as_primitive_array_unchecked().u32_iter())
            .unwrap()
            .flatten()
        {
            assert_eq!(o.unwrap(), 3);
        }
    }

    #[test]
    fn unary_fn_large_example() {
        let input = PrimitiveArray::from_vec(vec![2u32; 1025], Validity::AllValid);
        let output = input.unary(|u: u32| u + 1).unwrap();

        for o in output
            .with_dyn(|a| a.as_primitive_array_unchecked().u32_iter())
            .unwrap()
            .flatten()
        {
            assert_eq!(o.unwrap(), 3);
        }
    }

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
