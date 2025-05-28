use std::any::Any;

use num_traits::AsPrimitive;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValueType, NativeDecimalType, i256, match_each_decimal_value_type};

use crate::arrays::{BoolArray, DecimalArray};
use crate::builders::ArrayBuilder;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

#[derive(Debug)]
enum InnerDecimalBuilder {
    I8(BufferMut<i8>),
    I16(BufferMut<i16>),
    I32(BufferMut<i32>),
    I64(BufferMut<i64>),
    I128(BufferMut<i128>),
    I256(BufferMut<i256>),
}

macro_rules! impl_as_ref {
    ($typ:ty, $variant:path) => {
        impl AsRef<[$typ]> for InnerDecimalBuilder {
            fn as_ref(&self) -> &[$typ] {
                match self {
                    $variant(v) => v.as_ref(),
                    _ => vortex_panic!("Tried to access {} values from {self:?}", stringify!($typ)),
                }
            }
        }
    };
}

impl_as_ref!(i8, InnerDecimalBuilder::I8);
impl_as_ref!(i16, InnerDecimalBuilder::I16);
impl_as_ref!(i32, InnerDecimalBuilder::I32);
impl_as_ref!(i64, InnerDecimalBuilder::I64);
impl_as_ref!(i128, InnerDecimalBuilder::I128);
impl_as_ref!(i256, InnerDecimalBuilder::I256);

impl InnerDecimalBuilder {
    fn push_zeros(&mut self, n: usize) {
        match self {
            InnerDecimalBuilder::I8(v) => v.push_n(0, n),
            InnerDecimalBuilder::I16(v) => v.push_n(0, n),
            InnerDecimalBuilder::I32(v) => v.push_n(0, n),
            InnerDecimalBuilder::I64(v) => v.push_n(0, n),
            InnerDecimalBuilder::I128(v) => v.push_n(0, n),
            InnerDecimalBuilder::I256(v) => v.push_n(i256::ZERO, n),
        }
    }
    fn push<V: NativeDecimalType>(&mut self, value: V) {
        match self {
            InnerDecimalBuilder::I8(v) => v.push(value.as_()),
            InnerDecimalBuilder::I16(v) => v.push(value.as_()),
            InnerDecimalBuilder::I32(v) => v.push(value.as_()),
            InnerDecimalBuilder::I64(v) => v.push(value.as_()),
            InnerDecimalBuilder::I128(v) => v.push(value.as_()),
            InnerDecimalBuilder::I256(v) => v.push(value.as_()),
        }
    }

    fn len(&self) -> usize {
        match self {
            InnerDecimalBuilder::I8(v) => v.len(),
            InnerDecimalBuilder::I16(v) => v.len(),
            InnerDecimalBuilder::I32(v) => v.len(),
            InnerDecimalBuilder::I64(v) => v.len(),
            InnerDecimalBuilder::I128(v) => v.len(),
            InnerDecimalBuilder::I256(v) => v.len(),
        }
    }

    fn extend_from_buffer<T: NativeDecimalType>(&mut self, array: Buffer<T>) {
        match self {
            InnerDecimalBuilder::I8(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i8>>::as_(x)),
            ),
            InnerDecimalBuilder::I16(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i16>>::as_(x)),
            ),
            InnerDecimalBuilder::I32(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i32>>::as_(x)),
            ),
            InnerDecimalBuilder::I64(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i64>>::as_(x)),
            ),
            InnerDecimalBuilder::I128(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i128>>::as_(x)),
            ),
            InnerDecimalBuilder::I256(v) => v.extend(
                array
                    .iter()
                    .copied()
                    .map(|x| <T as AsPrimitive<i256>>::as_(x)),
            ),
        }
    }

    fn reserve(&mut self, n: usize) {
        match self {
            InnerDecimalBuilder::I8(v) => v.reserve(n),
            InnerDecimalBuilder::I16(v) => v.reserve(n),
            InnerDecimalBuilder::I32(v) => v.reserve(n),
            InnerDecimalBuilder::I64(v) => v.reserve(n),
            InnerDecimalBuilder::I128(v) => v.reserve(n),
            InnerDecimalBuilder::I256(v) => v.reserve(n),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            InnerDecimalBuilder::I8(v) => v.capacity(),
            InnerDecimalBuilder::I16(v) => v.capacity(),
            InnerDecimalBuilder::I32(v) => v.capacity(),
            InnerDecimalBuilder::I64(v) => v.capacity(),
            InnerDecimalBuilder::I128(v) => v.capacity(),
            InnerDecimalBuilder::I256(v) => v.capacity(),
        }
    }
}

macro_rules! impl_from_buffer {
    ($typ:ty, $variant:path) => {
        impl From<BufferMut<$typ>> for InnerDecimalBuilder {
            fn from(value: BufferMut<$typ>) -> Self {
                $variant(value)
            }
        }
    };
}

impl_from_buffer!(i8, InnerDecimalBuilder::I8);
impl_from_buffer!(i16, InnerDecimalBuilder::I16);
impl_from_buffer!(i32, InnerDecimalBuilder::I32);
impl_from_buffer!(i64, InnerDecimalBuilder::I64);
impl_from_buffer!(i128, InnerDecimalBuilder::I128);
impl_from_buffer!(i256, InnerDecimalBuilder::I256);

pub struct DecimalBuilder {
    values: InnerDecimalBuilder,
    nulls: LazyNullBufferBuilder,
    dtype: DType,
}

const DEFAULT_BUILDER_CAPACITY: usize = 1024;

impl DecimalBuilder {
    pub fn new<T: NativeDecimalType>(precision: u8, scale: i8, nullability: Nullability) -> Self {
        Self::with_capacity::<T>(
            DEFAULT_BUILDER_CAPACITY,
            DecimalDType::new(precision, scale),
            nullability,
        )
    }

    pub fn with_capacity<T: NativeDecimalType>(
        capacity: usize,
        decimal: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        let values: InnerDecimalBuilder = match T::VALUES_TYPE {
            DecimalValueType::I8 => BufferMut::<i8>::with_capacity(capacity).into(),
            DecimalValueType::I16 => BufferMut::<i16>::with_capacity(capacity).into(),
            DecimalValueType::I32 => BufferMut::<i32>::with_capacity(capacity).into(),
            DecimalValueType::I64 => BufferMut::<i64>::with_capacity(capacity).into(),
            DecimalValueType::I128 => BufferMut::<i128>::with_capacity(capacity).into(),
            DecimalValueType::I256 => BufferMut::<i256>::with_capacity(capacity).into(),
            v => unreachable!("invalid values type {v:?}"),
        };

        Self {
            values,
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Decimal(decimal, nullability),
        }
    }

    /// Append a `Mask` to the null buffer.
    pub fn append_mask(&mut self, mask: Mask) {
        self.nulls.append_validity_mask(mask);
    }

    fn extend_with_validity_mask(&mut self, validity_mask: Mask) {
        self.nulls.append_validity_mask(validity_mask);
    }

    pub fn append_value<V: NativeDecimalType>(&mut self, value: V) {
        self.values.push(value);
        self.nulls.append(true);
    }

    pub fn append_option<T>(&mut self, value: Option<T>)
    where
        Self: Send,
        T: NativeDecimalType + Default + Send + 'static,
    {
        match value {
            Some(value) => {
                self.values.push(value);
                self.nulls.append(true);
            }
            None => self.append_null(),
        }
    }

    pub fn finish_into_decimal(&mut self) -> DecimalArray {
        let nulls = self.nulls.finish();

        if let Some(null_buf) = nulls.as_ref() {
            assert_eq!(
                null_buf.len(),
                self.values.len(),
                "null buffer length must equal value buffer length"
            );
        }

        let validity = match (nulls, self.dtype.nullability()) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (Some(_), Nullability::NonNullable) => {
                vortex_panic!("Non-nullable builder has null values")
            }
            (None, Nullability::Nullable) => Validity::AllValid,
            (Some(nulls), Nullability::Nullable) => {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::Array(BoolArray::from(nulls.into_inner()).into_array())
                }
            }
        };

        let DType::Decimal(decimal_dtype, _) = self.dtype else {
            vortex_panic!("DecimalBuilder must have Decimal DType");
        };

        match &mut self.values {
            InnerDecimalBuilder::I8(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
            InnerDecimalBuilder::I16(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
            InnerDecimalBuilder::I32(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
            InnerDecimalBuilder::I64(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
            InnerDecimalBuilder::I128(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
            InnerDecimalBuilder::I256(v) => {
                DecimalArray::new(std::mem::take(v).freeze(), decimal_dtype, validity)
            }
        }
    }
}

impl ArrayBuilder for DecimalBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.push_zeros(n);
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.values.push_zeros(n);
        self.nulls.append_n_nulls(n);
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        let array = array.to_decimal()?;

        let DType::Decimal(decimal_dtype, _) = self.dtype else {
            vortex_panic!("DecimalBuilder must have Decimal DType");
        };

        if array.decimal_dtype() != decimal_dtype {
            vortex_bail!(
                "Cannot extend from array with different decimal type: {:?} != {:?}",
                array.decimal_dtype(),
                decimal_dtype
            );
        }

        match_each_decimal_value_type!(array.values_type(), |$D| {
            self.values.extend_from_buffer(array.buffer::<$D>())
        });

        self.extend_with_validity_mask(array.validity_mask()?);

        Ok(())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.values.capacity() {
            self.values.reserve(capacity - self.values.len());
            self.nulls.ensure_capacity(capacity);
        }
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_decimal().into_array()
    }
}

#[cfg(test)]
mod tests {
    use crate::builders::{ArrayBuilder, DecimalBuilder};

    #[test]
    fn test_mixed_extend() {
        let mut i8s = DecimalBuilder::new::<i8>(2, 1, false.into());
        i8s.append_value(10);
        i8s.append_value(11);
        i8s.append_value(12);
        let i8s = i8s.finish();

        let mut i128s = DecimalBuilder::new::<i128>(2, 1, false.into());
        i128s.extend_from_array(&i8s).unwrap();
        let i128s = i128s.finish_into_decimal();
        assert_eq!(i128s.buffer::<i128>().as_slice(), &[10, 11, 12]);
    }
}
