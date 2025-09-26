// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, IntegerPType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{ListScalar, Scalar};

use crate::arrays::ListArray;
use crate::builders::{
    ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder, PrimitiveBuilder,
    builder_with_capacity,
};
use crate::canonical::{Canonical, ToCanonical};
use crate::compute::cast;
use crate::{Array, ArrayRef, IntoArray};

/// The builder for building a [`ListArray`], parametrized by the `PType` of the offsets buffer.
pub struct ListBuilder<O: IntegerPType> {
    dtype: DType,
    /// The values of the list.
    value_builder: Box<dyn ArrayBuilder>,
    /// Represents the offsets into the values array.
    index_builder: PrimitiveBuilder<O>,
    nulls: LazyNullBufferBuilder,
}

impl<O: IntegerPType> ListBuilder<O> {
    /// Creates a new `ListBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(value_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::with_capacity(value_dtype, nullability, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `ListBuilder` with the given `capacity`.
    ///
    /// # Notes
    ///
    /// The number of indices is one more than the number of lists in the array!
    ///
    /// See also: [`ListBuilder::with_values_and_index_capacity`].
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        index_capacity: usize,
    ) -> Self {
        Self::with_values_and_index_capacity(
            value_dtype,
            nullability,
            // We choose an arbitrary capacity for values since we can't know the true capacity.
            2 * index_capacity,
            index_capacity,
        )
    }

    /// Create a ListBuilder with the specified capacity for indices and values.
    ///
    /// # Notes
    ///
    /// The number of indices is one more than the number of lists in the array!
    pub fn with_values_and_index_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        values_capacity: usize,
        index_capacity: usize,
    ) -> Self {
        let value_builder = builder_with_capacity(value_dtype.as_ref(), values_capacity);
        let mut index_builder = PrimitiveBuilder::<O>::with_capacity(NonNullable, index_capacity);

        // The first index of the list, which is always 0 and represents an empty list.
        index_builder.append_zero();

        Self {
            value_builder,
            index_builder,
            nulls: LazyNullBufferBuilder::new(index_capacity),
            dtype: DType::List(value_dtype, nullability),
        }
    }

    /// Appends a list `value` to the builder.
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        match value.elements() {
            None => {
                if self.dtype.nullability() == NonNullable {
                    vortex_bail!("Cannot append null value to non-nullable list");
                }
                self.append_null();
            }
            Some(elements) => {
                for scalar in elements {
                    // TODO(connor): This is slow, we should be able to append multiple values at
                    // once, or the list scalar should hold an Array
                    self.value_builder.append_scalar(&scalar)?;
                }

                self.nulls.append_non_null();
                self.index_builder.append_value(
                    O::from_usize(self.value_builder.len())
                        .vortex_expect("Failed to convert from usize to O"),
                );
            }
        }

        Ok(())
    }

    /// Finishes the builder directly into a [`ListArray`].
    pub fn finish_into_list(&mut self) -> ListArray {
        assert_eq!(
            self.index_builder.len(),
            self.nulls.len() + 1,
            "Indices length must be one more than nulls length."
        );

        ListArray::try_new(
            self.value_builder.finish(),
            self.index_builder.finish(),
            self.nulls.finish_with_nullability(self.dtype.nullability()),
        )
        .vortex_expect("Buffer, offsets, and validity must have same length.")
    }

    /// The [`DType`] of the inner elements. Note that this is **not** the same as the [`DType`] of
    /// the outer `List`.
    pub fn element_dtype(&self) -> &DType {
        let DType::List(element_dtype, _) = &self.dtype else {
            vortex_panic!("`ListBuilder` has an incorrect dtype: {}", self.dtype);
        };

        element_dtype
    }
}

impl<O: IntegerPType> ArrayBuilder for ListBuilder<O> {
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
        let curr_len = self.value_builder.len();
        for _ in 0..n {
            self.index_builder.append_value(
                O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            )
        }
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        let count = self.value_builder.len();
        for _ in 0..n {
            // A list with a null element is can be a list with a zero-span offset and a validity
            // bit set
            self.index_builder.append_value(
                O::from_usize(count).vortex_expect("Failed to convert from usize to <O>"),
            )
        }
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "ListBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        let list_scalar = ListScalar::try_from(scalar)?;
        self.append_value(list_scalar)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let list = array.to_list();
        if list.is_empty() {
            return;
        }

        let builder_len = self.value_builder.len();
        let builder_len_offset = match O::from_usize(builder_len) {
            Some(v) => v,
            None => {
                vortex_panic!(
                    "cannot convert length {} to type {:?}",
                    builder_len,
                    O::PTYPE
                )
            }
        };

        let offsets = list.offsets();
        let elements = list.elements();

        let index_dtype = self.index_builder.dtype();

        // Cast offsets to the correct type upfront.
        let casted_offsets =
            cast(offsets, index_dtype).vortex_expect("Offsets must be castable to index dtype");

        // Convert to primitive and get as slice.
        let offsets_primitive = casted_offsets.to_primitive();
        let offsets_slice = offsets_primitive.as_slice::<O>();

        // Get the first offset (leading junk values count).
        let n_leading_junk_values = offsets_slice[0];
        let n_leading_junk_values_usize: usize = n_leading_junk_values.as_();

        // Manually adjust offsets and append to index_builder.
        for i in 1..offsets_slice.len() {
            let offset = offsets_slice[i];
            let adjusted = offset - n_leading_junk_values + builder_len_offset;
            self.index_builder.append_value(adjusted);
        }

        // Extract non-junk values.
        let last_offset = offsets_slice[offsets_slice.len() - 1];
        let last_offset_usize: usize = last_offset.as_();
        let non_junk_values = elements.slice(n_leading_junk_values_usize..last_offset_usize);

        self.nulls.append_validity_mask(array.validity_mask());
        self.value_builder.ensure_capacity(non_junk_values.len());
        self.value_builder.extend_from_array(&non_junk_values);
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.value_builder.ensure_capacity(capacity);
        self.index_builder.ensure_capacity(capacity);
        self.nulls.ensure_capacity(capacity);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_list().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::List(self.finish_into_list())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use Nullability::{NonNullable, Nullable};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, IntegerPType, Nullability};
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::{ChunkedArray, ListArray};
    use crate::builders::ArrayBuilder;
    use crate::builders::list::ListBuilder;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray, ToCanonical};

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

        let list_array = list.to_list();

        assert_eq!(list_array.list_elements_at(0).len(), 3);
        assert_eq!(list_array.list_elements_at(1).len(), 3);
    }

    #[test]
    fn test_append_empty_list() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(dtype.clone(), NonNullable, 0);

        assert!(
            builder
                .append_value(Scalar::list_empty(dtype, NonNullable).as_list())
                .is_ok()
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

        let list_array = list.to_list();

        assert_eq!(list_array.list_elements_at(0).len(), 3);
        assert_eq!(list_array.list_elements_at(1).len(), 0);
        assert_eq!(list_array.list_elements_at(2).len(), 3);
    }

    fn test_extend_builder_gen<O: IntegerPType>() {
        let list = ListArray::from_iter_opt_slow::<O, _, _>(
            [Some(vec![0, 1, 2]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )
        .unwrap();

        let mut builder = ListBuilder::<O>::with_capacity(Arc::new(I32.into()), Nullable, 6);

        builder.extend_from_array(&list);
        builder.extend_from_array(&list);
        builder.extend_from_array(&list.slice(0..0));
        builder.extend_from_array(&list.slice(1..3));

        let expected = ListArray::from_iter_opt_slow::<O, _, _>(
            [
                Some(vec![0, 1, 2]),
                None,
                Some(vec![4, 5]),
                Some(vec![0, 1, 2]),
                None,
                Some(vec![4, 5]),
                None,
                Some(vec![4, 5]),
            ],
            Arc::new(DType::Primitive(I32, NonNullable)),
        )
        .unwrap()
        .to_list();

        let actual = builder.finish_into_canonical().into_list();

        assert_eq!(
            actual.elements().to_primitive().as_slice::<i32>(),
            expected.elements().to_primitive().as_slice::<i32>()
        );

        assert_eq!(
            actual.offsets().to_primitive().as_slice::<O>(),
            expected.offsets().to_primitive().as_slice::<O>()
        );

        assert_eq!(actual.validity(), expected.validity())
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

        let canon_values = chunked_list.unwrap().to_list();

        assert_eq!(
            one_trailing_unused_element.scalar_at(0),
            canon_values.scalar_at(0)
        );
        assert_eq!(second_array.scalar_at(0), canon_values.scalar_at(1));
    }

    #[test]
    fn test_append_scalar() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u64>::with_capacity(dtype.clone(), Nullable, 10);

        // Test appending a valid list.
        let list_scalar1 = Scalar::list(dtype.clone(), vec![1i32.into(), 2i32.into()], Nullable);
        builder.append_scalar(&list_scalar1).unwrap();

        // Test appending another list.
        let list_scalar2 = Scalar::list(
            dtype.clone(),
            vec![3i32.into(), 4i32.into(), 5i32.into()],
            Nullable,
        );
        builder.append_scalar(&list_scalar2).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(DType::List(dtype.clone(), Nullable));
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_list();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.

        let scalar0 = array.scalar_at(0);
        let list0 = scalar0.as_list();
        assert_eq!(list0.len(), 2);
        if let Some(list0_items) = list0.elements() {
            assert_eq!(list0_items[0].as_primitive().typed_value::<i32>(), Some(1));
            assert_eq!(list0_items[1].as_primitive().typed_value::<i32>(), Some(2));
        }

        let scalar1 = array.scalar_at(1);
        let list1 = scalar1.as_list();
        assert_eq!(list1.len(), 3);
        if let Some(list1_items) = list1.elements() {
            assert_eq!(list1_items[0].as_primitive().typed_value::<i32>(), Some(3));
            assert_eq!(list1_items[1].as_primitive().typed_value::<i32>(), Some(4));
            assert_eq!(list1_items[2].as_primitive().typed_value::<i32>(), Some(5));
        }

        let scalar2 = array.scalar_at(2);
        let list2 = scalar2.as_list();
        assert!(list2.is_null()); // This should be null.

        // Check validity.
        assert!(array.validity().is_valid(0));
        assert!(array.validity().is_valid(1));
        assert!(!array.validity().is_valid(2));

        // Test wrong dtype error.
        let mut builder = ListBuilder::<u64>::with_capacity(dtype, NonNullable, 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
