use std::any::Any;
use std::sync::Arc;

use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{ListScalar, NumericOperator};

use crate::arrays::{ConstantArray, ListArray, OffsetPType};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::{ArrayBuilder, ArrayBuilderExt, PrimitiveBuilder, builder_with_capacity};
use crate::compute::{cast, numeric};
use crate::{Array, ArrayRef, ToCanonical};

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

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        self.nulls.append_validity_mask(array.validity_mask()?);

        let list = array.to_list()?;

        let cursor_usize = self.value_builder.len();
        let cursor = O::from_usize(cursor_usize).ok_or_else(|| {
            vortex_err!(
                "cannot convert length {} to type {:?}",
                cursor_usize,
                O::PTYPE
            )
        })?;

        let offsets = numeric(
            &cast(
                &list.offsets().slice(1, list.offsets().len())?,
                &DType::Primitive(O::PTYPE, NonNullable),
            )?,
            &ConstantArray::new(cursor, list.len()),
            NumericOperator::Add,
        )?;
        self.index_builder.extend_from_array(&offsets)?;

        if !list.is_empty() {
            let last_used_index = self.index_builder.values().last().vortex_expect("there must be at least one index because we just extended a non-zero list of offsets");
            let sliced_values = list
                .elements()
                .slice(0, last_used_index.as_() - cursor_usize)?;
            self.value_builder.ensure_capacity(sliced_values.len());
            self.value_builder.extend_from_array(&sliced_values)?;
        }

        Ok(())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.index_builder.ensure_capacity(capacity);
        self.value_builder.ensure_capacity(capacity);
        self.nulls.ensure_capacity(capacity);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        assert_eq!(
            self.index_builder.len(),
            self.nulls.len() + 1,
            "Indices length must be one more than nulls length."
        );

        ListArray::try_new(
            self.value_builder.finish(),
            self.index_builder.finish(),
            self.nulls.finish_with_nullability(self.nullability),
        )
        .vortex_expect("Buffer, offsets, and validity must have same length.")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use Nullability::{NonNullable, Nullable};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::{ChunkedArray, ListArray, OffsetPType};
    use crate::builders::ArrayBuilder;
    use crate::builders::list::ListBuilder;
    use crate::compute::scalar_at;
    use crate::validity::Validity;
    use crate::{IntoArray as _, ToCanonical};

    #[test]
    fn test_empty() {
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), NonNullable, 0);

        let list = builder.finish();
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

        let list = builder.finish();
        assert_eq!(list.len(), 2);

        let list_array = list.to_list().unwrap();

        assert_eq!(list_array.elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.elements_at(1).unwrap().len(), 3);
    }

    #[test]
    fn test_non_null_fails() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), NonNullable, 0);

        assert!(
            builder
                .append_value(Scalar::list_empty(dtype, NonNullable).as_list())
                .is_err()
        )
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

        let list = builder.finish();
        assert_eq!(list.len(), 3);

        let list_array = list.to_list().unwrap();

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

        builder.extend_from_array(&list).unwrap();
        builder.extend_from_array(&list).unwrap();

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
        .to_list()
        .unwrap();

        let res = builder
            .finish()
            .to_canonical()
            .unwrap()
            .into_list()
            .unwrap();

        assert_eq!(
            res.elements().to_primitive().unwrap().as_slice::<i32>(),
            expect.elements().to_primitive().unwrap().as_slice::<i32>()
        );

        assert_eq!(
            res.offsets().to_primitive().unwrap().as_slice::<O>(),
            expect.offsets().to_primitive().unwrap().as_slice::<O>()
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

    #[test]
    pub fn test_array_with_gap() {
        let one_trailing_unused_element = ListArray::try_new(
            buffer![1, 2, 3, 4].into_array(),
            buffer![0, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let second_array = ListArray::try_new(
            buffer![5, 6].into_array(),
            buffer![0, 2].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let chunked_list = ChunkedArray::try_new(
            vec![
                one_trailing_unused_element.clone().into_array(),
                second_array.clone().into_array(),
            ],
            DType::List(Arc::new(DType::Primitive(I32, NonNullable)), NonNullable),
        );

        let canon_values = chunked_list.unwrap().to_list().unwrap();

        assert_eq!(
            scalar_at(&one_trailing_unused_element, 0).unwrap(),
            scalar_at(&canon_values, 0).unwrap()
        );
        assert_eq!(
            scalar_at(&second_array, 0).unwrap(),
            scalar_at(&canon_values, 1).unwrap()
        );
    }
}
