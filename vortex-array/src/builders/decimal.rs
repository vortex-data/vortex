// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::arrays::DecimalArray;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::match_each_decimal_value;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

/// The builder for building a [`DecimalArray`].
///
/// The output will be a new [`DecimalArray`] holding values of `T`. Any value that is a valid
/// [decimal type][NativeDecimalType] can be appended to the builder and it will be immediately
/// coerced into the target type.
pub struct DecimalBuilder {
    dtype: DType,
    values: DecimalBuffer,
    nulls: LazyBitBufferBuilder,
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
    pub fn new<T: NativeDecimalType>(decimal: DecimalDType, nullability: Nullability) -> Self {
        Self::with_capacity::<T>(DEFAULT_BUILDER_CAPACITY, decimal, nullability)
    }

    /// Creates a new `DecimalBuilder` with the given `capacity`.
    pub fn with_capacity<T: NativeDecimalType>(
        capacity: usize,
        decimal: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self {
            dtype: DType::Decimal(decimal, nullability),
            values: match_each_decimal_value_type!(T::DECIMAL_TYPE, |D| {
                DecimalBuffer::from(BufferMut::<D>::with_capacity(capacity))
            }),
            nulls: LazyBitBufferBuilder::new(capacity),
        }
    }

    /// Appends a decimal `value` to the builder.
    pub fn append_value<V: NativeDecimalType>(&mut self, value: V) {
        self.values.push(value);
        self.nulls.append_non_null();
    }

    /// Appends `n` copies of `value` as non-null entries, directly writing into the buffer.
    pub fn append_n_values<V: NativeDecimalType>(&mut self, value: V, n: usize) {
        self.values.push_n(value, n);
        self.nulls.append_n_non_nulls(n);
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
            "DecimalBuilder expected scalar with dtype {}, got {}",
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

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        #[expect(deprecated)]
        let decimal_array = array.to_decimal();

        match_each_decimal_value_type!(decimal_array.values_type(), |D| {
            // Extends the values buffer from another buffer of type D where D can be coerced to the
            // builder type.
            self.values
                .extend(decimal_array.buffer::<D>().iter().copied());
        });

        self.nulls.append_validity_mask(
            &decimal_array
                .as_ref()
                .validity()
                .vortex_expect("validity_mask")
                .execute_mask(
                    decimal_array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .vortex_expect("Failed to compute validity mask"),
        );
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::from_validity_mask(validity);
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
            buffer.push(
                <T as BigCast>::from(value)
                    .ok_or_else(|| {
                        vortex_err!(
                            "decimal conversion failure {:?}, type: {:?} to {:?}",
                            value,
                            V::DECIMAL_TYPE,
                            T::DECIMAL_TYPE,
                        )
                    })
                    .vortex_expect("operation should succeed in builder"),
            )
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
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::DecimalBuilder;
    use crate::builders::decimal::DecimalArray;
    use crate::dtype::DecimalDType;

    #[test]
    fn test_mixed_extend() {
        let values = 42i8;

        let mut i8s = DecimalBuilder::new::<i8>(DecimalDType::new(2, 1), false.into());
        for v in 0..values {
            i8s.append_value(v);
        }
        let i8s = i8s.finish();

        let mut i128s = DecimalBuilder::new::<i128>(DecimalDType::new(2, 1), false.into());
        i128s.extend_from_array(&i8s);
        let i128s = i128s.finish();

        for i in 0..i8s.len() {
            assert_eq!(
                i8s.execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap(),
                i128s
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_append_scalar() {
        use crate::scalar::Scalar;

        // Simply test that the builder accepts its own finish output via scalar.
        let mut builder = DecimalBuilder::new::<i64>(DecimalDType::new(10, 2), true.into());
        builder.append_value(1234i64);
        builder.append_value(5678i64);
        builder.append_null();

        let array = builder.finish();
        let expected = DecimalArray::from_option_iter(
            [Some(1234i64), Some(5678), None],
            DecimalDType::new(10, 2),
        );
        assert_arrays_eq!(&array, &expected);

        // Test by taking a scalar from the array and appending it to a new builder.
        let mut builder2 = DecimalBuilder::new::<i64>(DecimalDType::new(10, 2), true.into());
        for i in 0..array.len() {
            let scalar = array
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap();
            builder2.append_scalar(&scalar).unwrap();
        }

        let array2 = builder2.finish();
        assert_arrays_eq!(&array2, &array);

        // Test wrong dtype error.
        let mut builder = DecimalBuilder::new::<i64>(DecimalDType::new(10, 2), false.into());
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
