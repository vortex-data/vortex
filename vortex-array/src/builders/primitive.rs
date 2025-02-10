use std::any::Any;

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::AllOr;

use crate::array::{BoolArray, PrimitiveArray};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoCanonical};

pub struct PrimitiveBuilder<T: NativePType> {
    values: BufferMut<T>,
    nulls: LazyNullBufferBuilder,
    dtype: DType,
}

impl<T: NativePType> PrimitiveBuilder<T> {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append(true);
    }

    pub fn append_option(&mut self, value: Option<T>) {
        match value {
            Some(value) => {
                self.values.push(value);
                self.nulls.append(true);
            }
            None => self.append_null(),
        }
    }
}

impl<T: NativePType> ArrayBuilder for PrimitiveBuilder<T> {
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

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = array.into_canonical()?.into_primitive()?;
        if array.ptype() != T::PTYPE {
            vortex_bail!("Cannot extend from array with different ptype");
        }

        self.values.extend_from_slice(array.as_slice::<T>());

        match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                self.nulls.append_n_non_nulls(array.len());
            }
            AllOr::None => self.nulls.append_n_nulls(array.len()),
            AllOr::Some(validity) => self.nulls.append_buffer(validity.clone()),
        }

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        let validity = match (self.nulls.finish(), self.dtype().nullability()) {
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

        Ok(PrimitiveArray::new(std::mem::take(&mut self.values).freeze(), validity).into_array())
    }
}
