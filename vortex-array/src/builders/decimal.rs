// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{BigCast, NativeDecimalType, i256, match_each_decimal_value_type};

use crate::arrays::DecimalArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// The builder for building a [`DecimalArray`].
///
/// The output will be a new [`DecimalArray`] holding values of `T`. Any value that is a valid
/// [decimal type][NativeDecimalType] can be appended to the builder and it will be immediately
/// coerced into the target type.
pub struct DecimalBuilder {
    dtype: DType,
    values: DecimalBuffer,
    nulls: LazyNullBufferBuilder,
}

/// Wrapper around the typed builder.
///
/// We want to be able to downcast a `Box<dyn ArrayBuilder>` to a [`DecimalBuilder`] and we
/// generally don't have enough type information to get the `T` at the call site, so we instead use
/// this to hold values and can push values into the correct buffer type generically.
enum DecimalBuffer {
    I8(BufferMut<i8>),
    I16(BufferMut<i16>),
    I32(BufferMut<i32>),
    I64(BufferMut<i64>),
    I128(BufferMut<i128>),
    I256(BufferMut<i256>),
}

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

impl DecimalBuilder {
    /// Creates a new `DecimalBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new<T: NativeDecimalType>(precision: u8, scale: i8, nullability: Nullability) -> Self {
        Self::with_capacity::<T>(
            DEFAULT_BUILDER_CAPACITY,
            DecimalDType::new(precision, scale),
            nullability,
        )
    }

    /// Creates a new `DecimalBuilder` with the given `capacity`.
    pub fn with_capacity<T: NativeDecimalType>(
        capacity: usize,
        decimal: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self {
            dtype: DType::Decimal(decimal, nullability),
            values: match_each_decimal_value_type!(T::VALUES_TYPE, |D| {
                DecimalBuffer::from(BufferMut::<D>::with_capacity(capacity))
            }),
            nulls: LazyNullBufferBuilder::new(capacity),
        }
    }

    /// Appends a decimal `value` to the builder.
    pub fn append_value<V: NativeDecimalType>(&mut self, value: V) {
        self.values.push(value);
        self.nulls.append_non_null();
    }

    /// Finishes the builder directly into a [`DecimalArray`].
    pub fn finish_into_decimal(&mut self) -> DecimalArray {
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        let decimal_dtype = *self.decimal_dtype();

        delegate_fn!(std::mem::take(&mut self.values), |T, values| {
            DecimalArray::new::<T>(values.freeze(), decimal_dtype, validity)
        })
    }

    /// The [`DecimalDType`] of this builder.
    pub fn decimal_dtype(&self) -> &DecimalDType {
        let DType::Decimal(decimal_dtype, _) = &self.dtype else {
            vortex_panic!("`DecimalBuilder` somehow had dtype {}", self.dtype);
        };

        decimal_dtype
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
        if !self.dtype.eq_with_nullability_superset(array.dtype()) {
            vortex_bail!(
                "tried to extend a builder with `DType` {} with an array with `DType {}",
                self.dtype,
                array.dtype()
            );
        }

        let decimal_array = array
            .to_decimal()
            .vortex_expect("we checked that the array had `DType::Decimal`");

        match_each_decimal_value_type!(decimal_array.values_type(), |D| {
            // Extends the values buffer from another buffer of type D where D can be coerced to the
            // builder type.
            self.values
                .extend(decimal_array.buffer::<D>().iter().copied());
        });

        self.nulls
            .append_validity_mask(decimal_array.validity_mask());

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

macro_rules! impl_from_buffer {
    ($T:ty, $variant:ident) => {
        impl From<BufferMut<$T>> for DecimalBuffer {
            fn from(buffer: BufferMut<$T>) -> Self {
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

impl Default for DecimalBuffer {
    fn default() -> Self {
        Self::I8(BufferMut::<i8>::empty())
    }
}

#[cfg(test)]
mod tests {
    use crate::builders::{ArrayBuilder, DecimalBuilder};

    #[test]
    fn test_mixed_extend() {
        let values = 42i8;

        let mut i8s = DecimalBuilder::new::<i8>(2, 1, false.into());
        for v in 0..values {
            i8s.append_value(v);
        }
        let i8s = i8s.finish();

        let mut i128s = DecimalBuilder::new::<i128>(2, 1, false.into());
        i128s.extend_from_array(&i8s).unwrap();
        let i128s = i128s.finish();

        for i in 0..i8s.len() {
            assert_eq!(i8s.scalar_at(i), i128s.scalar_at(i));
        }
    }
}
