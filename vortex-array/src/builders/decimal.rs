use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::NativeDecimalType;

use crate::arrays::{BoolArray, DecimalArray};
use crate::builders::ArrayBuilder;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

pub struct DecimalBuilder<T> {
    values: BufferMut<T>,
    nulls: LazyNullBufferBuilder,
    dtype: DType,
}

const DEFAULT_BUILDER_CAPACITY: usize = 1024;

impl<T: NativeDecimalType> DecimalBuilder<T> {
    pub fn new(precision: u8, scale: i8, nullability: Nullability) -> Self {
        Self::with_capacity(
            DEFAULT_BUILDER_CAPACITY,
            DecimalDType::new(precision, scale),
            nullability,
        )
    }

    pub fn with_capacity(capacity: usize, decimal: DecimalDType, nullability: Nullability) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
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

    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append(true);
    }

    pub fn append_option(&mut self, value: Option<T>)
    where
        Self: Send,
        T: Default + Send + 'static,
    {
        match value {
            Some(value) => {
                self.values.push(value);
                self.nulls.append(true);
            }
            None => self.append_null(),
        }
    }

    pub fn values(&self) -> &[T] {
        self.values.as_ref()
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

        DecimalArray::new(
            std::mem::take(&mut self.values).freeze(),
            decimal_dtype,
            validity,
        )
    }
}

impl<T: NativeDecimalType + Default + Send + 'static> ArrayBuilder for DecimalBuilder<T> {
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
        self.values.push_n(T::default(), n);
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
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

        self.values
            .extend_from_slice(array.buffer::<T>().as_slice());
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
