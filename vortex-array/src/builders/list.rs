use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{ListScalar, Scalar};

use crate::array::ListArray;
use crate::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt, BoolBuilder};
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData};

pub struct ListBuilder {
    value_builder: Box<dyn ArrayBuilder>,
    index_builder: Box<dyn ArrayBuilder>,
    validity: BoolBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl ListBuilder {
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let value_builder = builder_with_capacity(value_dtype.as_ref(), capacity);
        let mut index_builder = if capacity < 2usize.pow(31) - 1 {
            builder_with_capacity(
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                capacity,
            )
        } else {
            builder_with_capacity(
                &DType::Primitive(PType::I64, Nullability::NonNullable),
                capacity,
            )
        };

        // The first index of the list, which is always 0 and represents an empty list.
        index_builder.append_zero();

        Self {
            value_builder,
            index_builder,
            validity: BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
            nullability,
            dtype: DType::List(value_dtype, nullability),
        }
    }

    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        let count = self.value_builder.len();
        if value.is_null() {
            self.append_null();
            Ok(())
        } else {
            for scalar in value.elements() {
                // TODO(joe): This is slow, we should be able to append multiple values at once,
                // or the list scalar should hold an ArrayData
                self.value_builder.append_scalar(&scalar)?;
            }
            self.index_builder
                .append_scalar(&Scalar::from(count + self.value_builder.len()))
        }
    }
}

impl ArrayBuilder for ListBuilder {
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
        self.validity.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.value_builder.append_zeros(n);
        let count = self.value_builder.len();
        for i in 0..n {
            self.index_builder
                .append_scalar(&Scalar::from(count + i + 1))
                .vortex_expect("Failed to append index");
        }
        self.validity.append_values(true, n);
    }

    fn append_nulls(&mut self, n: usize) {
        let count = self.value_builder.len();
        for _ in 0..n {
            // A list with a null element is can be a list with a zero-span offset and a validity
            // bit set
            self.index_builder
                .append_scalar(&Scalar::from(count))
                .vortex_expect("Failed to append null");
            self.validity.append_null();
        }
    }

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let validity = match self.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::Array(self.validity.finish()?),
        };

        ListArray::try_new(
            self.value_builder.finish()?,
            self.index_builder.finish()?,
            validity,
        )
        .map(ListArray::into_array)
    }
}
