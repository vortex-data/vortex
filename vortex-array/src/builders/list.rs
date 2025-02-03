use std::any::Any;
use std::sync::Arc;

use num_traits::{AsPrimitive, PrimInt};
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::{ListScalar, Scalar};

use crate::array::ListArray;
use crate::builders::{
    builder_with_capacity, ArrayBuilder, ArrayBuilderExt, BoolBuilder, PrimitiveBuilder,
};
use crate::validity::Validity;
use crate::{Array, IntoArray};

pub struct ListBuilder<O: PrimInt + NativePType> {
    value_builder: Box<dyn ArrayBuilder>,
    index_builder: PrimitiveBuilder<O>,
    validity: BoolBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl<O> ListBuilder<O>
where
    O: PrimInt + NativePType,
    Scalar: From<O>,
    usize: AsPrimitive<O>,
{
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        // I would expect the list to have more than one value per index
        let value_builder = builder_with_capacity(value_dtype.as_ref(), 2 * capacity);
        let mut index_builder = PrimitiveBuilder::with_capacity(Nullability::NonNullable, capacity);

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
        match value.elements() {
            None => {
                if self.nullability == Nullability::NonNullable {
                    vortex_bail!("Cannot append null value to non-nullable list");
                }
                self.append_null();
                Ok(())
            }
            Some(elements) => {
                for scalar in elements {
                    // TODO(joe): This is slow, we should be able to append multiple values at once,
                    // or the list scalar should hold an Array
                    self.value_builder.append_scalar(&scalar)?;
                }
                self.validity.append_value(true);
                self.append_index(self.value_builder.len().as_())
            }
        }
    }

    fn append_index(&mut self, index: O) -> VortexResult<()> {
        self.index_builder.append_scalar(&Scalar::from(index))
    }
}

impl<O> ArrayBuilder for ListBuilder<O>
where
    O: PrimInt + NativePType,
    Scalar: From<O>,
    usize: AsPrimitive<O>,
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
        self.validity.len()
    }

    fn append_zeros(&mut self, n: usize) {
        let count = self.value_builder.len();
        self.value_builder.append_zeros(n);
        for i in 0..n {
            self.append_index((count + i + 1).as_())
                .vortex_expect("Failed to append index");
        }
        self.validity.append_values(true, n);
    }

    fn append_nulls(&mut self, n: usize) {
        let count = self.value_builder.len();
        for _ in 0..n {
            // A list with a null element is can be a list with a zero-span offset and a validity
            // bit set
            self.append_index(count.as_())
                .vortex_expect("Failed to append index");
        }
        self.validity.append_values(false, n);
    }

    fn finish(&mut self) -> VortexResult<Array> {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;
    use Nullability::{NonNullable, Nullable};

    use crate::builders::list::ListBuilder;
    use crate::builders::ArrayBuilder;
    use crate::IntoArrayVariant;

    #[test]
    fn test_empty() {
        let mut builder =
            ListBuilder::<u32>::with_capacity(Arc::new(PType::I32.into()), NonNullable, 0);

        let list = builder.finish().unwrap();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_values() {
        let dtype: Arc<DType> = Arc::new(PType::I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), NonNullable, 0);

        builder
            .append_value(
                Scalar::list(
                    dtype.clone(),
                    vec![1i32.into(), 2i32.into(), 3i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        builder
            .append_value(
                Scalar::list(
                    dtype,
                    vec![4i32.into(), 5i32.into(), 6i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let list = builder.finish().unwrap();
        assert_eq!(list.len(), 2);

        let list_array = list.into_list().unwrap();

        assert_eq!(list_array.elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.elements_at(1).unwrap().len(), 3);
    }

    #[test]
    fn test_non_null_fails() {
        let dtype: Arc<DType> = Arc::new(PType::I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), NonNullable, 0);

        assert!(builder
            .append_value(Scalar::list_empty(dtype, NonNullable).as_list())
            .is_err())
    }

    #[test]
    fn test_nullable_values() {
        let dtype: Arc<DType> = Arc::new(PType::I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), Nullable, 0);

        builder
            .append_value(
                Scalar::list(
                    dtype.clone(),
                    vec![1i32.into(), 2i32.into(), 3i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        builder
            .append_value(Scalar::list_empty(dtype.clone(), NonNullable).as_list())
            .unwrap();

        builder
            .append_value(
                Scalar::list(
                    dtype,
                    vec![4i32.into(), 5i32.into(), 6i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let list = builder.finish().unwrap();
        assert_eq!(list.len(), 3);

        let list_array = list.into_list().unwrap();

        assert_eq!(list_array.elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.elements_at(1).unwrap().len(), 0);
        assert_eq!(list_array.elements_at(2).unwrap().len(), 3);
    }
}
