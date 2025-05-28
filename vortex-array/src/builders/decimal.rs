use std::any::Any;

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{BigCast, NativeDecimalType, i256, match_each_decimal_value_type};

use crate::arrays::{BoolArray, DecimalArray};
use crate::builders::ArrayBuilder;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

// Wrapper around the typed builder.
// We want to be able to downcast a Box<dyn ArrayBuilder> to a DecimalBuilder and we generally
// don't have enough type information to get the `T` at the call site, so we instead use this
// to hold values and can push values into the correct buffer type generically.
enum DecimalBuffer {
    I8(BufferMut<i8>),
    I16(BufferMut<i16>),
    I32(BufferMut<i32>),
    I64(BufferMut<i64>),
    I128(BufferMut<i128>),
    I256(BufferMut<i256>),
}

impl Default for DecimalBuffer {
    fn default() -> Self {
        Self::I8(BufferMut::<i8>::empty())
    }
}

macro_rules! impl_from_buffer {
    ($typ:ty, $variant:ident) => {
        impl From<BufferMut<$typ>> for DecimalBuffer {
            fn from(buffer: BufferMut<$typ>) -> Self {
                Self::$variant(buffer)
            }
        }
    };
}
impl_from_buffer!(i8, I8);
impl_from_buffer!(i16, I16);
impl_from_buffer!(i32, I32);
impl_from_buffer!(i64, I64);
impl_from_buffer!(i128, I128);
impl_from_buffer!(i256, I256);

macro_rules! delegate_fn {
    ($self:expr, | $tname:ident, $buffer:ident | $body:block) => {{
        #[allow(unused)]
        match $self {
            DecimalBuffer::I8(buffer) => {
                type $tname = i8;
                let $buffer = buffer;
                $body
            }
            DecimalBuffer::I16(buffer) => {
                type $tname = i16;
                let $buffer = buffer;
                $body
            }
            DecimalBuffer::I32(buffer) => {
                type $tname = i32;
                let $buffer = buffer;
                $body
            }
            DecimalBuffer::I64(buffer) => {
                type $tname = i64;
                let $buffer = buffer;
                $body
            }
            DecimalBuffer::I128(buffer) => {
                type $tname = i128;
                let $buffer = buffer;
                $body
            }
            DecimalBuffer::I256(buffer) => {
                type $tname = i256;
                let $buffer = buffer;
                $body
            }
        }
    }};
}

impl DecimalBuffer {
    fn push<V: NativeDecimalType>(&mut self, value: V) {
        delegate_fn!(self, |T, buffer| {
            buffer.push(<T as BigCast>::from(value).vortex_expect("decimal conversion failure"))
        });
    }

    fn push_n<V: NativeDecimalType>(&mut self, value: V, n: usize) {
        delegate_fn!(self, |T, buffer| {
            buffer.push_n(
                <T as BigCast>::from(value).vortex_expect("decimal conversion failure"),
                n,
            )
        });
    }

    fn reserve(&mut self, additional: usize) {
        delegate_fn!(self, |T, buffer| { buffer.reserve(additional) })
    }

    fn capacity(&self) -> usize {
        delegate_fn!(self, |T, buffer| { buffer.capacity() })
    }

    fn len(&self) -> usize {
        delegate_fn!(self, |T, buffer| { buffer.len() })
    }

    pub fn extend<I, V: NativeDecimalType>(&mut self, iter: I)
    where
        I: Iterator<Item = V>,
    {
        delegate_fn!(self, |T, buffer| {
            buffer.extend(
                iter.map(|x| <T as BigCast>::from(x).vortex_expect("decimal conversion failure")),
            )
        })
    }
}

/// An [`ArrayBuilder`] for `Decimal` typed arrays.
///
/// The output will be a new [`DecimalArray`] holding values of `T`. Any value that is
/// a valid [decimal type][NativeDecimalType] can be appended to the builder and it will be
/// immediately coerced into the target type.
pub struct DecimalBuilder {
    values: DecimalBuffer,
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
        Self {
            values: match_each_decimal_value_type!(T::VALUES_TYPE, |$D| {
                DecimalBuffer::from(BufferMut::<$D>::with_capacity(capacity))
            }),
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Decimal(decimal, nullability),
        }
    }
}

impl DecimalBuilder {
    fn extend_with_validity_mask(&mut self, validity_mask: Mask) {
        self.nulls.append_validity_mask(validity_mask);
    }

    /// Extend the values buffer from another buffer of type V where V can be coerced
    /// to the builder type.
    fn extend_from_buffer<V: NativeDecimalType>(&mut self, values: &Buffer<V>) {
        self.values.extend(values.iter().copied());
    }
}

impl DecimalBuilder {
    pub fn append_value<V: NativeDecimalType>(&mut self, value: V) {
        self.values.push(value);
        self.nulls.append(true);
    }

    pub fn append_option<V: NativeDecimalType>(&mut self, value: Option<V>) {
        match value {
            Some(value) => {
                self.values.push(value);
                self.nulls.append(true);
            }
            None => self.append_null(),
        }
    }

    /// Append a `Mask` to the null buffer.
    pub fn append_mask(&mut self, mask: Mask) {
        self.nulls.append_validity_mask(mask);
    }
}

impl DecimalBuilder {
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

        delegate_fn!(std::mem::take(&mut self.values), |T, values| {
            DecimalArray::new::<T>(values.freeze(), decimal_dtype, validity)
        })
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
        self.values.push_n(0, n);
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.values.push_n(0, n);
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
            self.extend_from_buffer(&array.buffer::<$D>())
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
