use std::any::Any;

use arrow_array::builder::{ArrayBuilder as _, BooleanBuilder as ArrowBooleanBuilder};
use arrow_array::Array;
use vortex_dtype::Nullability;
use vortex_error::{vortex_bail, VortexResult};

use crate::arrow::FromArrowArray;
use crate::builders::ArrayBuilder;
use crate::ArrayData;

pub struct BoolBuilder {
    inner: ArrowBooleanBuilder,
    nullability: Nullability,
    // TODO(ngates): track stats?
}

impl BoolBuilder {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: ArrowBooleanBuilder::with_capacity(capacity),
            nullability,
        }
    }

    pub fn append_value(&mut self, value: bool) {
        self.inner.append_value(value);
    }

    pub fn append_values(&mut self, value: bool, n: usize) {
        self.inner.append_n(n, value);
    }

    pub fn append_option(&mut self, value: Option<bool>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }
}

impl ArrayBuilder for BoolBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.inner.append_n(n, false);
    }

    fn append_nulls(&mut self, n: usize) {
        self.inner.append_nulls(n);
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let arrow = self.inner.finish();

        if self.nullability == Nullability::NonNullable && arrow.null_count() > 0 {
            vortex_bail!("Non-nullable builder has null values");
        }

        Ok(ArrayData::from_arrow(&arrow, self.nullability.into()))
    }
}
