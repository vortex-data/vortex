// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::RangeInclusive;

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::cast_value_in_precision;
use crate::arrays::decimal::precision_bounds;
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

/// Builds a [`DecimalArray`] at the canonical width for its precision.
///
/// Appended values may use any [`NativeDecimalType`]. The builder converts them to the canonical
/// width before storing them.
pub struct DecimalBuilder {
    dtype: DType,
    values: DecimalBuffer,
    nulls: LazyBitBufferBuilder,
}

/// Stores values and their precision bounds at the canonical width.
enum DecimalBuffer {
    I8(DecimalValues<i8>),
    I16(DecimalValues<i16>),
    I32(DecimalValues<i32>),
    I64(DecimalValues<i64>),
    I128(DecimalValues<i128>),
    I256(DecimalValues<i256>),
}

struct DecimalValues<T> {
    buffer: BufferMut<T>,
    bounds: RangeInclusive<T>,
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
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(decimal: DecimalDType, nullability: Nullability) -> Self {
        Self::new_in(decimal, nullability, BufferAllocatorRef::static_ref())
    }

    /// Creates a new `DecimalBuilder` with the default capacity using `allocator`.
    pub fn new_in(
        decimal: DecimalDType,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_in(DEFAULT_BUILDER_CAPACITY, decimal, nullability, allocator)
    }

    /// Creates a new `DecimalBuilder` with the given `capacity`.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(capacity: usize, decimal: DecimalDType, nullability: Nullability) -> Self {
        Self::with_capacity_in(
            capacity,
            decimal,
            nullability,
            BufferAllocatorRef::static_ref(),
        )
    }

    /// Creates a new `DecimalBuilder` with `capacity` using `allocator`.
    pub fn with_capacity_in(
        capacity: usize,
        decimal: DecimalDType,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self {
            dtype: DType::Decimal(decimal, nullability),
            values: match_each_decimal_value_type!(decimal.canonical_values_type(), |D| {
                DecimalBuffer::from(DecimalValues {
                    buffer: BufferMut::<D>::with_capacity_in(capacity, allocator.clone()),
                    bounds: precision_bounds::<D>(decimal),
                })
            }),
            nulls: LazyBitBufferBuilder::new(capacity, allocator.clone()),
        }
    }

    /// Appends one unscaled decimal value.
    ///
    /// # Panics
    ///
    /// Panics if `value` falls outside the declared precision.
    pub fn append_value<V: NativeDecimalType>(&mut self, value: V) {
        let dtype = *self.decimal_dtype();
        delegate_fn!(&mut self.values, |T, values| {
            let value = cast_value_in_precision(value, dtype, &values.bounds);
            values.buffer.push(value);
        });
        self.nulls.append_non_null();
    }

    /// Appends `n` copies of an unscaled decimal value.
    ///
    /// # Panics
    ///
    /// Panics if `value` falls outside the declared precision.
    pub fn append_n_values<V: NativeDecimalType>(&mut self, value: V, n: usize) {
        let dtype = *self.decimal_dtype();
        delegate_fn!(&mut self.values, |T, values| {
            let value = cast_value_in_precision(value, dtype, &values.bounds);
            values.buffer.push_n(value, n);
        });
        self.nulls.append_n_non_nulls(n);
    }

    /// Appends a canonical [`DecimalArray`] of the builder's dtype.
    pub(crate) fn append_decimal_array(
        &mut self,
        array: ArrayView<'_, Decimal>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(array.dtype(), self.dtype());

        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        delegate_fn!(&mut self.values, |T, values| {
            values.buffer.extend_from_slice(&array.buffer::<T>());
        });
        self.nulls.append_validity_mask(&mask);
        Ok(())
    }

    /// Finishes the builder directly into a [`DecimalArray`].
    pub fn finish_into_decimal(&mut self) -> DecimalArray {
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        let decimal_dtype = *self.decimal_dtype();

        delegate_fn!(&mut self.values, |T, values| {
            let values = values.buffer.take();
            assert!(
                validity.maybe_len().is_none_or(|len| len == values.len()),
                "validity of length {:?} does not cover the {} values appended",
                validity.maybe_len(),
                values.len()
            );

            // SAFETY: The builder allocated a typed buffer at the dtype's canonical width, and
            // the length check above covers array-backed validity.
            unsafe { DecimalArray::new_unchecked::<T>(values.freeze(), decimal_dtype, validity) }
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
        self.values.push_zeros(n);
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.push_zeros(n);
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "DecimalBuilder received a scalar with the wrong dtype"
        );

        match scalar.as_decimal().decimal_value() {
            None => self.append_null(),
            Some(value) => {
                match_each_decimal_value!(value, |v| { self.values.push_scalar(v) });
                self.nulls.append_non_null();
            }
        }

        Ok(())
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_decimal().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Decimal(self.finish_into_decimal())
    }
}

impl DecimalBuffer {
    fn push_scalar<V: NativeDecimalType>(&mut self, value: V) {
        delegate_fn!(self, |T, values| {
            let value = <T as BigCast>::from(value)
                .vortex_expect("a scalar of the builder dtype fits its canonical width");
            values.buffer.push(value);
        });
    }

    fn push_zeros(&mut self, n: usize) {
        delegate_fn!(self, |T, values| { values.buffer.push_n(T::default(), n) });
    }

    fn reserve(&mut self, additional: usize) {
        delegate_fn!(self, |T, values| { values.buffer.reserve(additional) })
    }

    fn len(&self) -> usize {
        delegate_fn!(self, |T, values| { values.buffer.len() })
    }
}

macro_rules! impl_from_values {
    ($T:ty, $variant:ident) => {
        impl From<DecimalValues<$T>> for DecimalBuffer {
            fn from(values: DecimalValues<$T>) -> Self {
                Self::$variant(values)
            }
        }
    };
}

impl_from_values!(i8, I8);
impl_from_values!(i16, I16);
impl_from_values!(i32, I32);
impl_from_values!(i64, I64);
impl_from_values!(i128, I128);
impl_from_values!(i256, I256);

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::VortexSessionExecute;
    use crate::array::VTable;
    use crate::arrays::Decimal;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::DecimalBuilder;
    use crate::builders::decimal::DecimalArray;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::single(|builder: &mut DecimalBuilder| builder.append_value(-100i64))]
    #[case::repeated(|builder: &mut DecimalBuilder| builder.append_n_values(100i64, 3))]
    fn invalid_append_preserves_the_builder(#[case] append: fn(&mut DecimalBuilder)) {
        let dtype = DecimalDType::new(2, 1);
        let mut builder = DecimalBuilder::new_in(
            dtype,
            Nullability::Nullable,
            BufferAllocatorRef::static_ref(),
        );
        builder.append_null();

        assert!(catch_unwind(AssertUnwindSafe(|| append(&mut builder))).is_err());
        builder.append_value(99i64);
        assert_arrays_eq!(
            builder.finish(),
            DecimalArray::from_option_iter([None, Some(99i8)], dtype),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn append_values_at_canonical_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(2, 1);
        let mut builder = DecimalBuilder::new_in(
            dtype,
            Nullability::Nullable,
            BufferAllocatorRef::static_ref(),
        );
        builder.append_value(-42i128);
        builder.append_n_values(99i16, 2);
        builder.append_null();
        let array = builder.finish_into_decimal();
        assert_eq!(array.values_type(), DecimalType::I8);

        array.append_to_builder(&mut builder, &mut ctx)?;
        assert_arrays_eq!(
            builder.finish(),
            DecimalArray::from_option_iter([Some(-42i8), Some(99), Some(99), None], dtype),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn append_array_rejects_mismatched_dtype() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([1i8], DecimalDType::new(2, 0));
        let mut builder = DecimalBuilder::new_in(
            DecimalDType::new(4, 0),
            Nullability::NonNullable,
            BufferAllocatorRef::static_ref(),
        );

        assert!(Decimal::append_to_builder(array.as_view(), &mut builder, &mut ctx).is_err());
        assert!(builder.is_empty());
    }

    #[test]
    fn test_append_scalar() {
        let mut ctx = SESSION.create_execution_ctx();

        // Simply test that the builder accepts its own finish output via scalar.
        let mut builder = DecimalBuilder::new_in(
            DecimalDType::new(10, 2),
            true.into(),
            BufferAllocatorRef::static_ref(),
        );
        builder.append_value(1234i64);
        builder.append_value(5678i64);
        builder.append_null();

        let array = builder.finish();
        let expected = DecimalArray::from_option_iter(
            [Some(1234i64), Some(5678), None],
            DecimalDType::new(10, 2),
        );
        assert_arrays_eq!(&array, &expected, &mut ctx);

        // Test by taking a scalar from the array and appending it to a new builder.
        let mut builder2 = DecimalBuilder::new_in(
            DecimalDType::new(10, 2),
            true.into(),
            BufferAllocatorRef::static_ref(),
        );
        for i in 0..array.len() {
            let scalar = array
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .unwrap();
            builder2.append_scalar(&scalar).unwrap();
        }

        let array2 = builder2.finish();
        assert_arrays_eq!(&array2, &array, &mut ctx);

        // Test wrong dtype error.
        let mut builder = DecimalBuilder::new_in(
            DecimalDType::new(10, 2),
            false.into(),
            BufferAllocatorRef::static_ref(),
        );
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
