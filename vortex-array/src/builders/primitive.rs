use std::any::Any;

use arrow_array::builder::{
    ArrayBuilder as ArrowArrayBuilder, PrimitiveBuilder as ArrowPrimitiveBuilder,
};
use arrow_array::{Array, ArrowPrimitiveType};
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::arrow::FromArrowArray;
use crate::builders::ArrayBuilder;
use crate::ArrayData;

pub struct PrimitiveBuilder<T: NativePType> {
    inner: ArrowPrimitiveBuilder<T::ArrowPrimitiveType>,
    dtype: DType,
}

impl<T: NativePType + 'static> PrimitiveBuilder<T>
where
    T: NativePType,
    <T::ArrowPrimitiveType as ArrowPrimitiveType>::Native: NativePType,
{
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: ArrowPrimitiveBuilder::<T::ArrowPrimitiveType>::with_capacity(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    pub fn append_value(
        &mut self,
        value: <<T as NativePType>::ArrowPrimitiveType as ArrowPrimitiveType>::Native,
    ) {
        self.inner.append_value(value);
    }

    pub fn append_option(
        &mut self,
        value: Option<<<T as NativePType>::ArrowPrimitiveType as ArrowPrimitiveType>::Native>,
    ) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }
}

impl<T> ArrayBuilder for PrimitiveBuilder<T>
where
    T: NativePType + 'static,
    <T::ArrowPrimitiveType as ArrowPrimitiveType>::Native: NativePType,
{
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
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.inner.append_value_n(
            <<T::ArrowPrimitiveType as ArrowPrimitiveType>::Native>::default(),
            n,
        );
    }

    fn append_nulls(&mut self, n: usize) {
        self.inner.append_nulls(n);
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let arrow = self.inner.finish();

        if !self.dtype().is_nullable() && arrow.null_count() > 0 {
            vortex_bail!("Non-nullable builder {} has null values", self.dtype());
        }

        Ok(ArrayData::from_arrow(&arrow, self.dtype.is_nullable()))
    }
}
