// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::ListArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::builders::PrimitiveBuilder;
use crate::builders::builder_with_capacity;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

/// The builder for building a [`ListArray`], parametrized by the [`IntegerPType`] of the `offsets`
/// builder.
pub struct ListBuilder<O: IntegerPType> {
    /// The [`DType`] of the [`ListArray`]. This **must** be a [`DType::List`].
    dtype: DType,

    /// The builder for the underlying elements of the [`ListArray`].
    elements_builder: Box<dyn ArrayBuilder>,

    /// The builder for the `offsets` into the `elements` array.
    offsets_builder: PrimitiveBuilder<O>,

    /// The null map builder of the [`ListArray`].
    nulls: LazyBitBufferBuilder,
}

impl<O: IntegerPType> ListBuilder<O> {
    /// Creates a new `ListBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(value_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::with_capacity(
            value_dtype,
            nullability,
            // We arbitrarily choose 2 times the number of list scalars for the capacity of the
            // elements builder since we cannot know this ahead of time.
            DEFAULT_BUILDER_CAPACITY * 2,
            DEFAULT_BUILDER_CAPACITY,
        )
    }

    /// Create a new [`ListArray`] builder with a with the given `capacity`, as well as an initial
    /// capacity for the `elements` builder (since we cannot know that ahead of time solely based on
    /// the outer array `capacity`).
    ///
    /// # Notes
    ///
    /// The number of offsets is one more than the length (# of list scalars) in the array.
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        elements_capacity: usize,
        capacity: usize,
    ) -> Self {
        let elements_builder = builder_with_capacity(value_dtype.as_ref(), elements_capacity);
        let mut offsets_builder = PrimitiveBuilder::<O>::with_capacity(NonNullable, capacity + 1);

        // The first offset is always 0 and represents an empty list.
        offsets_builder.append_zero();

        Self {
            elements_builder,
            offsets_builder,
            nulls: LazyBitBufferBuilder::new(capacity),
            dtype: DType::List(value_dtype, nullability),
        }
    }

    /// Appends an array as a single non-null list entry to the builder.
    ///
    /// The input `array` must have the same dtype as the element dtype of this list builder.
    ///
    /// Note that the list entry will be non-null but the elements themselves are allowed to be null
    /// (only if the elements [`DType`] in nullable, of course).
    pub fn append_array_as_list(&mut self, array: &ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == self.element_dtype(),
            "Array dtype {:?} does not match list element dtype {:?}",
            array.dtype(),
            self.element_dtype()
        );

        self.elements_builder.extend_from_array(array);
        self.nulls.append_non_null();
        self.offsets_builder.append_value(
            O::from_usize(self.elements_builder.len())
                .vortex_expect("Failed to convert from usize to O"),
        );

        Ok(())
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
                    self.elements_builder.append_scalar(&scalar)?;
                }

                self.nulls.append_non_null();
                self.offsets_builder.append_value(
                    O::from_usize(self.elements_builder.len())
                        .vortex_expect("Failed to convert from usize to O"),
                );
            }
        }

        Ok(())
    }

    /// Finishes the builder directly into a [`ListArray`].
    pub fn finish_into_list(&mut self) -> ListArray {
        assert_eq!(
            self.offsets_builder.len(),
            self.nulls.len() + 1,
            "offsets length must be one more than nulls length."
        );

        ListArray::try_new(
            self.elements_builder.finish(),
            self.offsets_builder.finish(),
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
        let curr_len = self.elements_builder.len();
        for _ in 0..n {
            self.offsets_builder.append_value(
                O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            )
        }
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        let curr_len = self.elements_builder.len();
        for _ in 0..n {
            // A list with a null element is can be a list with a zero-span offset and a validity
            // bit set
            self.offsets_builder.append_value(
                O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            )
        }
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "ListBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        self.append_value(scalar.as_list())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        let list = array.to_listview();
        if list.is_empty() {
            return;
        }

        // Append validity information.
        self.nulls.append_validity_mask(
            array
                .validity()
                .vortex_expect("validity_mask in extend_from_array_unchecked")
                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("Failed to compute validity mask"),
        );

        // Note that `ListViewArray` has `n` offsets and sizes, not `n+1` offsets like `ListArray`.
        let elements = list.elements();
        let offsets = list.offsets().to_primitive();
        let sizes = list.sizes().to_primitive();

        fn extend_inner<O, OffsetType, SizeType>(
            builder: &mut ListBuilder<O>,
            new_elements: &ArrayRef,
            new_offsets: &[OffsetType],
            new_sizes: &[SizeType],
        ) where
            O: IntegerPType,
            OffsetType: IntegerPType,
            SizeType: IntegerPType,
        {
            let num_lists = new_offsets.len();
            debug_assert_eq!(num_lists, new_sizes.len());

            let mut curr_offset = builder.elements_builder.len();
            let mut offsets_range = builder.offsets_builder.uninit_range(num_lists);

            // We need to append each list individually, converting from `ListViewArray` format to
            // the `ListArray` format that `ListBuilder` expects.
            for i in 0..new_offsets.len() {
                let offset: usize = new_offsets[i].as_();
                let size: usize = new_sizes[i].as_();

                if size > 0 {
                    let list_elements = new_elements
                        .slice(offset..offset + size)
                        .vortex_expect("list builder slice");
                    builder.elements_builder.extend_from_array(&list_elements);
                    curr_offset += size;
                }

                let new_offset =
                    O::from_usize(curr_offset).vortex_expect("Failed to convert offset");

                offsets_range.set_value(i, new_offset);
            }

            // SAFETY: We have initialized all `num_lists` values, and since the `offsets` array is
            // non-nullable, we are done.
            unsafe { offsets_range.finish() };
        }

        match_each_integer_ptype!(offsets.ptype(), |OffsetType| {
            match_each_integer_ptype!(sizes.ptype(), |SizeType| {
                extend_inner(
                    self,
                    elements,
                    offsets.as_slice::<OffsetType>(),
                    sizes.as_slice::<SizeType>(),
                )
            })
        })
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.elements_builder.reserve_exact(additional);
        self.offsets_builder.reserve_exact(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_list().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::List(self.finish_into_list().into_array().to_listview())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use Nullability::NonNullable;
    use Nullability::Nullable;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::ToCanonical;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::list::ListArrayExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::list::ListArray;
    use crate::builders::list::ListBuilder;
    use crate::dtype::DType;
    use crate::dtype::IntegerPType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::executor::VortexSessionExecute;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn test_empty() {
        let mut builder =
            ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), NonNullable, 0, 0);

        let list = builder.finish();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_values() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), NonNullable, 0, 0);

        builder
            .append_value(
                Scalar::list(
                    Arc::clone(&dtype),
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

        let list_array = list.to_listview();

        assert_eq!(list_array.list_elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.list_elements_at(1).unwrap().len(), 3);
    }

    #[test]
    fn test_append_empty_list() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), NonNullable, 0, 0);

        assert!(
            builder
                .append_value(Scalar::list_empty(dtype, NonNullable).as_list())
                .is_ok()
        )
    }

    #[test]
    fn test_nullable_values() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), Nullable, 0, 0);

        builder
            .append_value(
                Scalar::list(
                    Arc::clone(&dtype),
                    vec![1i32.into(), 2i32.into(), 3i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        builder
            .append_value(Scalar::list_empty(Arc::clone(&dtype), NonNullable).as_list())
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

        let list_array = list.to_listview();

        assert_eq!(list_array.list_elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.list_elements_at(1).unwrap().len(), 0);
        assert_eq!(list_array.list_elements_at(2).unwrap().len(), 3);
    }

    fn test_extend_builder_gen<O: IntegerPType>() {
        let list = ListArray::from_iter_opt_slow::<O, _, _>(
            [Some(vec![0, 1, 2]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )
        .unwrap();
        assert_eq!(list.len(), 3);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let mut builder = ListBuilder::<O>::with_capacity(Arc::new(I32.into()), Nullable, 18, 9);
        builder.extend_from_array(&list);
        builder.extend_from_array(&list);
        builder.extend_from_array(&list.slice(0..0).unwrap());
        builder.extend_from_array(&list.slice(1..3).unwrap());

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
        .to_listview();

        let actual = builder.finish_into_canonical().into_listview();

        assert_arrays_eq!(actual.elements(), expected.elements());

        assert_arrays_eq!(actual.offsets(), expected.offsets());

        assert!(
            actual
                .validity()
                .vortex_expect("list validity should be derivable")
                .mask_eq(
                    &expected
                        .validity()
                        .vortex_expect("list validity should be derivable"),
                    &mut ctx,
                )
                .unwrap(),
        );
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

        let canon_values = chunked_list.unwrap().as_array().to_listview();

        assert_eq!(
            one_trailing_unused_element.scalar_at(0).unwrap(),
            canon_values.scalar_at(0).unwrap()
        );
        assert_eq!(
            second_array.scalar_at(0).unwrap(),
            canon_values.scalar_at(1).unwrap()
        );
    }

    #[test]
    fn test_append_scalar() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListBuilder::<u64>::with_capacity(Arc::clone(&dtype), Nullable, 20, 10);

        // Test appending a valid list.
        let list_scalar1 =
            Scalar::list(Arc::clone(&dtype), vec![1i32.into(), 2i32.into()], Nullable);
        builder.append_scalar(&list_scalar1).unwrap();

        // Test appending another list.
        let list_scalar2 = Scalar::list(
            Arc::clone(&dtype),
            vec![3i32.into(), 4i32.into(), 5i32.into()],
            Nullable,
        );
        builder.append_scalar(&list_scalar2).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(DType::List(Arc::clone(&dtype), Nullable));
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_list();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.

        let scalar0 = array.scalar_at(0).unwrap();
        let list0 = scalar0.as_list();
        assert_eq!(list0.len(), 2);
        if let Some(list0_items) = list0.elements() {
            assert_eq!(list0_items[0].as_primitive().typed_value::<i32>(), Some(1));
            assert_eq!(list0_items[1].as_primitive().typed_value::<i32>(), Some(2));
        }

        let scalar1 = array.scalar_at(1).unwrap();
        let list1 = scalar1.as_list();
        assert_eq!(list1.len(), 3);
        if let Some(list1_items) = list1.elements() {
            assert_eq!(list1_items[0].as_primitive().typed_value::<i32>(), Some(3));
            assert_eq!(list1_items[1].as_primitive().typed_value::<i32>(), Some(4));
            assert_eq!(list1_items[2].as_primitive().typed_value::<i32>(), Some(5));
        }

        let scalar2 = array.scalar_at(2).unwrap();
        let list2 = scalar2.as_list();
        assert!(list2.is_null()); // This should be null.

        // Check validity.
        assert!(
            array
                .validity()
                .vortex_expect("list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            array
                .validity()
                .vortex_expect("list validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            !array
                .validity()
                .vortex_expect("list validity should be derivable")
                .is_valid(2)
                .unwrap()
        );

        // Test wrong dtype error.
        let mut builder = ListBuilder::<u64>::with_capacity(dtype, NonNullable, 20, 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }

    #[test]
    fn test_append_array_as_list() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), NonNullable, 20, 10);

        // Append a primitive array as a single list entry.
        let arr1 = buffer![1i32, 2, 3].into_array();
        builder.append_array_as_list(&arr1).unwrap();

        // Interleave with a list scalar.
        builder
            .append_value(
                Scalar::list(
                    Arc::clone(&dtype),
                    vec![10i32.into(), 11i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        // Append another primitive array as a single list entry.
        let arr2 = buffer![4i32, 5].into_array();
        builder.append_array_as_list(&arr2).unwrap();

        // Append an empty array as a single list entry (empty list).
        let arr3 = buffer![0i32; 0].into_array();
        builder.append_array_as_list(&arr3).unwrap();

        // Interleave with another list scalar (empty list).
        builder
            .append_value(Scalar::list_empty(Arc::clone(&dtype), NonNullable).as_list())
            .unwrap();

        let list = builder.finish_into_list();
        assert_eq!(list.len(), 5);

        // Verify elements array: [1, 2, 3, 10, 11, 4, 5].
        assert_arrays_eq!(
            list.elements(),
            PrimitiveArray::from_iter([1i32, 2, 3, 10, 11, 4, 5])
        );

        // Verify offsets array.
        assert_arrays_eq!(
            list.offsets(),
            PrimitiveArray::from_iter([0u32, 3, 5, 7, 7, 7])
        );

        // Test dtype mismatch error.
        let mut builder = ListBuilder::<u32>::with_capacity(dtype, NonNullable, 20, 10);
        let wrong_dtype_arr = buffer![1i64, 2, 3].into_array();
        assert!(builder.append_array_as_list(&wrong_dtype_arr).is_err());
    }
}
