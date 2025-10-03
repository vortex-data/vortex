// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{
    BigCast, DecimalValue, NativeDecimalType, Scalar, i256, match_each_decimal_value,
    match_each_decimal_value_type,
};

use crate::arrays::DecimalArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder};
use crate::canonical::Canonical;
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

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.push_n(0, n);
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "DecimalBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        match scalar.as_decimal().decimal_value() {
            None => self.append_null(),
            Some(v) => match_each_decimal_value!(v, |dec_val| {
                self.append_value(dec_val);
            }),
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let decimal_array = array.to_decimal();

        match_each_decimal_value_type!(decimal_array.values_type(), |D| {
            // Extends the values buffer from another buffer of type D where D can be coerced to the
            // builder type.
            self.values
                .extend(decimal_array.buffer::<D>().iter().copied());
        });

        self.nulls
            .append_validity_mask(decimal_array.validity_mask());
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.values.capacity() {
            self.values.reserve(capacity - self.values.len());
            self.nulls.ensure_capacity(capacity);
        }
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_decimal().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Decimal(self.finish_into_decimal())
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
        i128s.extend_from_array(&i8s);
        let i128s = i128s.finish();

        for i in 0..i8s.len() {
            assert_eq!(i8s.scalar_at(i), i128s.scalar_at(i));
        }
    }

    #[test]
    fn test_append_scalar() {
        use vortex_scalar::Scalar;

        // Simply test that the builder accepts its own finish output via scalar.
        let mut builder = DecimalBuilder::new::<i64>(10, 2, true.into());
        builder.append_value(1234i64);
        builder.append_value(5678i64);
        builder.append_null();

        let array = builder.finish();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.
        let scalar0 = array.scalar_at(0);
        let decimal0 = scalar0.as_decimal();
        assert!(decimal0.decimal_value().is_some());
        // We can't easily check the exact value without accessing internals.

        let scalar1 = array.scalar_at(1);
        let decimal1 = scalar1.as_decimal();
        assert!(decimal1.decimal_value().is_some());

        let scalar2 = array.scalar_at(2);
        let decimal2 = scalar2.as_decimal();
        assert!(decimal2.decimal_value().is_none()); // This should be null.

        // Test by taking a scalar from the array and appending it to a new builder.
        let mut builder2 = DecimalBuilder::new::<i64>(10, 2, true.into());
        for i in 0..array.len() {
            let scalar = array.scalar_at(i);
            builder2.append_scalar(&scalar).unwrap();
        }

        let array2 = builder2.finish();
        assert_eq!(array2.len(), 3);

        // Verify the values match.
        for i in 0..3 {
            assert_eq!(array.scalar_at(i), array2.scalar_at(i));
        }

        // Test wrong dtype error.
        let mut builder = DecimalBuilder::new::<i64>(10, 2, false.into());
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
