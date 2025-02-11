use std::any::Any;
use std::sync::Arc;

use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_mask::AllOr;
use vortex_scalar::{BinaryNumericOperator, ListScalar};

use crate::array::{ConstantArray, ListArray, OffsetPType};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt, PrimitiveBuilder};
use crate::compute::{binary_numeric, slice, try_cast};
use crate::{Array, IntoArray, IntoCanonical};

pub struct ListBuilder<O: NativePType> {
    value_builder: Box<dyn ArrayBuilder>,
    index_builder: PrimitiveBuilder<O>,
    nulls: LazyNullBufferBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl<O: OffsetPType> ListBuilder<O> {
    // TODO(joe): add value + index capacity ctor.
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        // I would expect the list to have more than one value per index
        let value_builder = builder_with_capacity(value_dtype.as_ref(), 2 * capacity);
        let mut index_builder = PrimitiveBuilder::with_capacity(NonNullable, capacity);

        // The first index of the list, which is always 0 and represents an empty list.
        index_builder.append_zero();

        Self {
            value_builder,
            index_builder,
            nulls: LazyNullBufferBuilder::new(capacity),
            nullability,
            dtype: DType::List(value_dtype, nullability),
        }
    }

    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        match value.elements() {
            None => {
                if self.nullability == NonNullable {
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
                self.nulls.append_non_null();
                self.append_index(
                    O::from_usize(self.value_builder.len())
                        .vortex_expect("Failed to convert from usize to O"),
                )
            }
        }
    }

    fn append_index(&mut self, index: O) -> VortexResult<()> {
        self.index_builder.append_scalar(&index.into())
    }
}

impl<O: OffsetPType> ArrayBuilder for ListBuilder<O> {
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
        self.nulls.len()
    }

    fn append_zeros(&mut self, n: usize) {
        let count = self.value_builder.len();
        self.value_builder.append_zeros(n);
        for i in 0..n {
            self.append_index(
                O::from_usize(count + i + 1).vortex_expect("Failed to convert from usize to <O>"),
            )
            .vortex_expect("Failed to append index");
        }
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        let count = self.value_builder.len();
        for _ in 0..n {
            // A list with a null element is can be a list with a zero-span offset and a validity
            // bit set
            self.append_index(
                O::from_usize(count).vortex_expect("Failed to convert from usize to <O>"),
            )
            .vortex_expect("Failed to append index");
        }
        self.nulls.append_n_nulls(n);
    }

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                self.nulls.append_n_non_nulls(array.len());
            }
            AllOr::None => self.nulls.append_n_nulls(array.len()),
            AllOr::Some(validity) => self.nulls.append_buffer(validity.clone()),
        }

        let list = array.into_canonical()?.into_list()?;

        let offset = self.value_builder.len();
        self.value_builder.extend_from_array(list.elements())?;

        let offsets = binary_numeric(
            &try_cast(
                slice(list.offsets(), 1, list.offsets().len())?,
                &DType::Primitive(O::PTYPE, NonNullable),
            )?,
            &ConstantArray::new(
                O::from_usize(offset).ok_or_else(|| {
                    vortex_err!("cannot convert offset {} to type {:?}", offset, O::PTYPE)
                })?,
                list.len(),
            ),
            BinaryNumericOperator::Add,
        )?;
        self.index_builder.extend_from_array(offsets)?;

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        ListArray::try_new(
            self.value_builder.finish()?,
            self.index_builder.finish()?,
            self.nulls.finish_with_nullability(self.nullability)?,
        )
        .map(ListArray::into_array)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;
    use Nullability::{NonNullable, Nullable};

    use crate::array::{ListArray, OffsetPType};
    use crate::builders::list::ListBuilder;
    use crate::builders::ArrayBuilder;
    use crate::{IntoArrayVariant, IntoCanonical};

    #[test]
    fn test_empty() {
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), NonNullable, 0);

        let list = builder.finish().unwrap();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_values() {
        let dtype: Arc<DType> = Arc::new(I32.into());
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
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), NonNullable, 0);

        assert!(builder
            .append_value(Scalar::list_empty(dtype, NonNullable).as_list())
            .is_err())
    }

    #[test]
    fn test_nullable_values() {
        let dtype: Arc<DType> = Arc::new(I32.into());
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

    fn test_extend_builder_gen<O: OffsetPType>() {
        let list = ListArray::from_iter_opt_slow::<O, _, _>(
            [Some(vec![0, 1, 2]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )
        .unwrap();

        let mut builder = ListBuilder::<O>::with_capacity(Arc::new(I32.into()), Nullable, 6);

        builder.extend_from_array(list.clone()).unwrap();
        builder.extend_from_array(list).unwrap();

        let expect = ListArray::from_iter_opt_slow::<O, _, _>(
            [
                Some(vec![0, 1, 2]),
                None,
                Some(vec![4, 5]),
                Some(vec![0, 1, 2]),
                None,
                Some(vec![4, 5]),
            ],
            Arc::new(DType::Primitive(I32, NonNullable)),
        )
        .unwrap()
        .into_list()
        .unwrap();

        let res = builder
            .finish()
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_list()
            .unwrap();

        assert_eq!(
            res.elements().into_primitive().unwrap().as_slice::<i32>(),
            expect
                .elements()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>()
        );

        assert_eq!(
            res.offsets().into_primitive().unwrap().as_slice::<O>(),
            expect.offsets().into_primitive().unwrap().as_slice::<O>()
        );

        assert_eq!(res.validity(), expect.validity())
    }

    #[test]
    fn test_extend_builder() {
        test_extend_builder_gen::<i8>();
        test_extend_builder_gen::<i16>();
        test_extend_builder_gen::<i32>();
        test_extend_builder_gen::<i64>();

        test_extend_builder_gen::<u8>();
        test_extend_builder_gen::<u16>();
        test_extend_builder_gen::<u32>();
        test_extend_builder_gen::<u64>();
    }
}
