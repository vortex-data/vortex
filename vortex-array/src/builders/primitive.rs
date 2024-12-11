use std::any::Any;
use std::iter;

use arrow_buffer::NullBufferBuilder;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{BoolArray, PrimitiveArray};
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData};

pub struct PrimitiveBuilder<T: NativePType> {
    values: Vec<T>,
    validity: NullBufferBuilder,
    dtype: DType,
}

impl<T: NativePType + 'static> PrimitiveBuilder<T> {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            validity: NullBufferBuilder::new(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.validity.append(true);
    }

    pub fn append_option(&mut self, value: Option<T>) {
        match value {
            Some(value) => {
                self.values.push(value);
                self.validity.append(true);
            }
            None => self.append_null(),
        }
    }
}

impl<T: NativePType + 'static> ArrayBuilder for PrimitiveBuilder<T> {
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
        self.values.extend(iter::repeat(T::default()).take(n));
        self.validity.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.values.extend(iter::repeat(T::default()).take(n));
        self.validity.append_n_nulls(n);
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let validity = match (self.validity.finish(), self.dtype().nullability()) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (Some(_), Nullability::NonNullable) => {
                vortex_bail!("Non-nullable builder has null values")
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

        Ok(PrimitiveArray::from_vec(std::mem::take(&mut self.values), validity).into_array())
    }
}
