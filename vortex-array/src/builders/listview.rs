// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ListView Builder Implementation.
//!
//! A builder for [`ListViewArray`] that tracks both offsets and sizes.
//!
//! Unlike [`ListArray`] which only tracks offsets, [`ListViewArray`] stores both offsets and sizes
//! in separate arrays for better compression.

use std::sync::Arc;

use vortex_dtype::{DType, IntegerPType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{ListScalar, Scalar};

use super::lazy_null_builder::LazyNullBufferBuilder;
use crate::array::{Array, ArrayRef, IntoArray};
use crate::arrays::{ListViewArray, list_view_from_list};
use crate::builders::{
    ArrayBuilder, DEFAULT_BUILDER_CAPACITY, PrimitiveBuilder, builder_with_capacity,
};
use crate::{Canonical, ToCanonical};

/// A builder for creating [`ListViewArray`] instances, parameterized by the [`IntegerPType`] of
/// the `offsets` and the `sizes` builders.
///
/// This builder tracks both offsets and sizes using potentially different integer types for memory
/// efficiency. For example, you might use `u64` for offsets but only `u8` for sizes if your lists
/// are small.
///
/// Any combination of [`IntegerPType`] types are valid, as long as the type of `sizes` can fit into
/// the type of `offsets`.
pub struct ListViewBuilder<O: IntegerPType, S: IntegerPType> {
    /// The [`DType`] of the [`ListViewArray`]. This **must** be a [`DType::List`].
    dtype: DType,

    /// The builder for the underlying elements of the [`ListArray`].
    elements_builder: Box<dyn ArrayBuilder>,

    /// The builder for the `offsets` into the `elements` array.
    offsets_builder: PrimitiveBuilder<O>,

    /// The builder for the `sizes` of each list view.
    sizes_builder: PrimitiveBuilder<S>,

    /// The null map builder of the [`ListViewArray`].
    nulls: LazyNullBufferBuilder,
}

impl<O: IntegerPType, S: IntegerPType> ListViewBuilder<O, S> {
    /// Creates a new `ListViewBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::with_capacity(
            element_dtype,
            nullability,
            // We arbitrarily choose 2 times the number of list scalars for the capacity of the
            // elements builder since we cannot know this ahead of time.
            DEFAULT_BUILDER_CAPACITY * 2,
            DEFAULT_BUILDER_CAPACITY,
        )
    }

    /// Create a new [`ListViewArray`] builder with a with the given `capacity`, as well as an
    /// initial capacity for the `elements` builder (since we cannot know that ahead of time solely
    /// based on the outer array `capacity`).
    ///
    /// # Panics
    ///
    /// Panics if the size type `S` cannot fit within the offset type `O`.
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        nullability: Nullability,
        elements_capacity: usize,
        capacity: usize,
    ) -> Self {
        // Validate that size type's maximum value fits within offset type's maximum value.
        // Since offsets are non-negative, we only need to check max values.
        assert!(
            S::max_value_as_u64() <= O::max_value_as_u64(),
            "Size type {:?} (max offset {}) must fit within offset type {:?} (max offset {})",
            S::PTYPE,
            S::max_value_as_u64(),
            O::PTYPE,
            O::max_value_as_u64()
        );

        let elements_builder = builder_with_capacity(&element_dtype, elements_capacity);

        let offsets_builder =
            PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, capacity);
        let sizes_builder =
            PrimitiveBuilder::<S>::with_capacity(Nullability::NonNullable, capacity);

        let nulls = LazyNullBufferBuilder::new(capacity);

        Self {
            dtype: DType::List(element_dtype, nullability),
            elements_builder,
            offsets_builder,
            sizes_builder,
            nulls,
        }
    }

    // TODO(connor): This should probably take a `&ListScalar` instead.
    /// Append a list of values to the builder.
    ///
    /// This method extends the value builder with the provided values and records
    /// the offset and size of the new list.
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        let Some(elements) = value.elements() else {
            // If `elements` is `None`, then the `value` is a null value.
            vortex_ensure!(
                self.dtype.is_nullable(),
                "Cannot append null value to non-nullable list builder"
            );
            self.append_null();
            return Ok(());
        };

        let curr_offset = self.elements_builder.len();
        let num_elements = elements.len();

        for scalar in elements {
            // TODO(connor): This is slow, we should be able to append multiple values at once, or
            // the list scalar should hold an Array
            self.elements_builder.append_scalar(&scalar)?;
        }
        self.nulls.append_non_null();

        self.offsets_builder.append_value(
            O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
        );
        self.sizes_builder.append_value(
            S::from_usize(num_elements).vortex_expect("Failed to convert from usize to `S`"),
        );

        Ok(())
    }

    /// Finishes the builder directly into a [`ListViewArray`].
    pub fn finish_into_listview(&mut self) -> ListViewArray {
        debug_assert_eq!(self.offsets_builder.len(), self.sizes_builder.len());
        debug_assert_eq!(self.offsets_builder.len(), self.nulls.len());

        let elements = self.elements_builder.finish();
        let offsets = self.offsets_builder.finish();
        let sizes = self.sizes_builder.finish();
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        ListViewArray::try_new(elements, offsets, sizes, validity)
            .vortex_expect("Failed to create ListViewArray")
    }

    /// The [`DType`] of the inner elements. Note that this is **not** the same as the [`DType`] of
    /// the outer `FixedSizeList`.
    pub fn element_dtype(&self) -> &DType {
        let DType::List(element_dtype, ..) = &self.dtype else {
            vortex_panic!("`ListViewBuilder` has an incorrect dtype: {}", self.dtype);
        };

        element_dtype
    }
}

impl<O: IntegerPType, S: IntegerPType> ArrayBuilder for ListViewBuilder<O, S> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.offsets_builder.len()
    }

    fn append_zeros(&mut self, n: usize) {
        debug_assert_eq!(self.offsets_builder.len(), self.sizes_builder.len());
        debug_assert_eq!(self.offsets_builder.len(), self.nulls.len());

        // Get the current position in the elements array.
        let curr_offset = self.elements_builder.len();

        // Since we consider the "zero" element of a list an empty list, we simply update the
        // `offsets` and `sizes` metadata to add an empty list.
        for _ in 0..n {
            self.offsets_builder.append_value(
                O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
            );
            self.sizes_builder.append_value(S::zero());
        }

        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        debug_assert_eq!(self.offsets_builder.len(), self.sizes_builder.len());
        debug_assert_eq!(self.offsets_builder.len(), self.nulls.len());

        // Get the current position in the elements array.
        let curr_offset = self.elements_builder.len();

        // A null list can have any representation, but we choose to use the zero representation.
        for _ in 0..n {
            self.offsets_builder.append_value(
                O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
            );
            self.sizes_builder.append_value(S::zero());
        }

        // This is the only difference from `append_zeros`.
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "ListViewBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        let list_scalar = scalar.as_list();
        self.append_value(list_scalar)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let list_array = array.to_list();
        if list_array.is_empty() {
            return;
        }

        // TODO(connor)[ListView]: fix this after list view is canonical
        let listview_array = list_view_from_list(list_array);

        // We assume the worst case scenario, where the list view array is stored completely out of
        // order, with many out-of-order offsets, and lots of garbage data. Thus, we simply iterate
        // over all of the lists in the array and copy the data into this builder.
        for i in 0..listview_array.len() {
            let list = listview_array.scalar_at(i);

            self.append_scalar(&list)
                .vortex_expect("was unable to extend the `ListViewBuilder`")
        }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.elements_builder.ensure_capacity(capacity * 2);
        self.offsets_builder.ensure_capacity(capacity);
        self.sizes_builder.ensure_capacity(capacity);
        self.nulls.ensure_capacity(capacity);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_listview().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        // TODO(connor)[ListView]: fix this after list view is canonical
        unimplemented!("TODO(connor)[ListView]: fix this after list view is canonical")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_dtype::PType::I32;
    use vortex_scalar::Scalar;

    use super::ListViewBuilder;
    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::ListArray;
    use crate::builders::ArrayBuilder;
    use crate::vtable::ValidityHelper;

    #[test]
    fn test_empty() {
        let mut builder =
            ListViewBuilder::<u32, u32>::with_capacity(Arc::new(I32.into()), NonNullable, 0, 0);

        let listview = builder.finish();
        assert_eq!(listview.len(), 0);
    }

    #[test]
    fn test_basic_append_and_nulls() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity(dtype.clone(), Nullable, 0, 0);

        // Append a regular list.
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

        // Append an empty list.
        builder
            .append_value(Scalar::list_empty(dtype.clone(), NonNullable).as_list())
            .unwrap();

        // Append a null list.
        builder.append_null();

        // Append another regular list.
        builder
            .append_value(
                Scalar::list(dtype, vec![4i32.into(), 5i32.into()], NonNullable).as_list(),
            )
            .unwrap();

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 4);

        // Check first list: [1, 2, 3].
        let first_list = listview.list_elements_at(0);
        assert_eq!(first_list.len(), 3);
        assert_eq!(first_list.scalar_at(0), 1i32.into());
        assert_eq!(first_list.scalar_at(1), 2i32.into());
        assert_eq!(first_list.scalar_at(2), 3i32.into());

        // Check empty list.
        let empty_list = listview.list_elements_at(1);
        assert_eq!(empty_list.len(), 0);

        // Check null list.
        assert!(!listview.validity().is_valid(2));

        // Check last list: [4, 5].
        let last_list = listview.list_elements_at(3);
        assert_eq!(last_list.len(), 2);
        assert_eq!(last_list.scalar_at(0), 4i32.into());
        assert_eq!(last_list.scalar_at(1), 5i32.into());
    }

    #[test]
    fn test_different_offset_size_types() {
        // Test u32 offsets with u8 sizes.
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            ListViewBuilder::<u32, u8>::with_capacity(dtype.clone(), NonNullable, 0, 0);

        builder
            .append_value(
                Scalar::list(dtype.clone(), vec![1i32.into(), 2i32.into()], NonNullable).as_list(),
            )
            .unwrap();

        builder
            .append_value(
                Scalar::list(
                    dtype,
                    vec![3i32.into(), 4i32.into(), 5i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 2);

        // Verify first list: [1, 2].
        let first = listview.list_elements_at(0);
        assert_eq!(first.scalar_at(0), 1i32.into());
        assert_eq!(first.scalar_at(1), 2i32.into());

        // Verify second list: [3, 4, 5].
        let second = listview.list_elements_at(1);
        assert_eq!(second.scalar_at(0), 3i32.into());
        assert_eq!(second.scalar_at(1), 4i32.into());
        assert_eq!(second.scalar_at(2), 5i32.into());

        // Test u64 offsets with u16 sizes.
        let dtype2: Arc<DType> = Arc::new(I32.into());
        let mut builder2 =
            ListViewBuilder::<u64, u16>::with_capacity(dtype2.clone(), NonNullable, 0, 0);

        for i in 0..5 {
            builder2
                .append_value(
                    Scalar::list(dtype2.clone(), vec![(i * 10).into()], NonNullable).as_list(),
                )
                .unwrap();
        }

        let listview2 = builder2.finish_into_listview();
        assert_eq!(listview2.len(), 5);

        // Verify the values: [0], [10], [20], [30], [40].
        for i in 0..5i32 {
            let list = listview2.list_elements_at(i as usize);
            assert_eq!(list.len(), 1);
            assert_eq!(list.scalar_at(0), (i * 10).into());
        }
    }

    #[test]
    fn test_builder_trait_methods() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity(dtype.clone(), Nullable, 0, 0);

        // Test append_zeros (creates empty lists).
        builder.append_zeros(2);
        assert_eq!(builder.len(), 2);

        // Test append_nulls.
        unsafe {
            builder.append_nulls_unchecked(2);
        }
        assert_eq!(builder.len(), 4);

        // Test append_scalar.
        let list_scalar = Scalar::list(dtype, vec![10i32.into(), 20i32.into()], Nullable);
        builder.append_scalar(&list_scalar).unwrap();
        assert_eq!(builder.len(), 5);

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 5);

        // First two are empty lists (from append_zeros).
        assert_eq!(listview.list_elements_at(0).len(), 0);
        assert_eq!(listview.list_elements_at(1).len(), 0);

        // Next two are nulls.
        assert!(!listview.validity().is_valid(2));
        assert!(!listview.validity().is_valid(3));

        // Last is the regular list: [10, 20].
        let last_list = listview.list_elements_at(4);
        assert_eq!(last_list.len(), 2);
        assert_eq!(last_list.scalar_at(0), 10i32.into());
        assert_eq!(last_list.scalar_at(1), 20i32.into());
    }

    #[test]
    fn test_extend_from_array() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create a source ListArray.
        let source = ListArray::from_iter_opt_slow::<u32, _, Vec<i32>>(
            [Some(vec![1, 2, 3]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )
        .unwrap();

        let mut builder = ListViewBuilder::<u32, u32>::with_capacity(dtype.clone(), Nullable, 0, 0);

        // Add initial data.
        builder
            .append_value(Scalar::list(dtype, vec![0i32.into()], NonNullable).as_list())
            .unwrap();

        // Extend from the ListArray.
        unsafe {
            builder.extend_from_array_unchecked(&source.into_array());
        }

        // Extend from empty array (should be no-op).
        let empty_source = ListArray::from_iter_opt_slow::<u32, _, Vec<i32>>(
            std::iter::empty::<Option<Vec<i32>>>(),
            Arc::new(I32.into()),
        )
        .unwrap();
        unsafe {
            builder.extend_from_array_unchecked(&empty_source.into_array());
        }

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 4);

        // Check the extended data.
        // First list: [0] (initial data).
        let first = listview.list_elements_at(0);
        assert_eq!(first.len(), 1);
        assert_eq!(first.scalar_at(0), 0i32.into());

        // Second list: [1, 2, 3] (from source).
        let second = listview.list_elements_at(1);
        assert_eq!(second.len(), 3);
        assert_eq!(second.scalar_at(0), 1i32.into());
        assert_eq!(second.scalar_at(1), 2i32.into());
        assert_eq!(second.scalar_at(2), 3i32.into());

        // Third list: null (from source).
        assert!(!listview.validity().is_valid(2));

        // Fourth list: [4, 5] (from source).
        let fourth = listview.list_elements_at(3);
        assert_eq!(fourth.len(), 2);
        assert_eq!(fourth.scalar_at(0), 4i32.into());
        assert_eq!(fourth.scalar_at(1), 5i32.into());
    }

    #[test]
    fn test_error_append_null_to_non_nullable() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            ListViewBuilder::<u32, u32>::with_capacity(dtype.clone(), NonNullable, 0, 0);

        // Create a null list with nullable type (since Scalar::null requires nullable type).
        let null_scalar = Scalar::null(DType::List(dtype, Nullable));
        let null_list = null_scalar.as_list();

        // This should fail because we're trying to append a null to a non-nullable builder.
        let result = builder.append_value(null_list);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("null value to non-nullable")
        );
    }

    #[test]
    #[should_panic(
        expected = "Size type I32 (max offset 2147483647) must fit within offset type I16 (max offset 32767)"
    )]
    fn test_error_invalid_type_combination() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        // This should panic because i32 (4 bytes) cannot fit within i16 (2 bytes).
        let _builder = ListViewBuilder::<i16, i32>::with_capacity(dtype, NonNullable, 0, 0);
    }
}
