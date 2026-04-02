// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ListView Builder Implementation.
//!
//! A builder for [`ListViewArray`] that tracks both offsets and sizes.
//!
//! Unlike [`ListArray`] which only tracks offsets, [`ListViewArray`] stores both offsets and sizes
//! in separate arrays for better compression.
//!
//! [`ListArray`]: crate::arrays::ListArray

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ToCanonical;
use crate::array::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::PrimitiveBuilder;
use crate::builders::UninitRange;
use crate::builders::builder_with_capacity;
use crate::builders::lazy_null_builder::LazyBitBufferBuilder;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

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

    /// The builder for the underlying elements of the [`ListArray`](crate::arrays::ListArray).
    elements_builder: Box<dyn ArrayBuilder>,

    /// The builder for the `offsets` into the `elements` array.
    offsets_builder: PrimitiveBuilder<O>,

    /// The builder for the `sizes` of each list view.
    sizes_builder: PrimitiveBuilder<S>,

    /// The null map builder of the [`ListViewArray`].
    nulls: LazyBitBufferBuilder,
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
        let elements_builder = builder_with_capacity(&element_dtype, elements_capacity);

        let offsets_builder =
            PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, capacity);
        let sizes_builder =
            PrimitiveBuilder::<S>::with_capacity(Nullability::NonNullable, capacity);

        let nulls = LazyBitBufferBuilder::new(capacity);

        Self {
            dtype: DType::List(element_dtype, nullability),
            elements_builder,
            offsets_builder,
            sizes_builder,
            nulls,
        }
    }

    /// Appends an array as a single non-null list entry to the builder.
    ///
    /// The input `array` must have the same dtype as the element dtype of this list builder.
    ///
    /// Note that the list entry will be non-null but the elements themselves are allowed to be null
    /// (only if the elements [`DType`] is nullable, of course).
    pub fn append_array_as_list(&mut self, array: &ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == self.element_dtype(),
            "Array dtype {:?} does not match list element dtype {:?}",
            array.dtype(),
            self.element_dtype()
        );

        let curr_offset = self.elements_builder.len();
        let num_elements = array.len();

        // We must assert this even in release mode to ensure that the safety comment in
        // `finish_into_listview` is correct.
        assert!(
            ((curr_offset + num_elements) as u64) < O::max_value_as_u64(),
            "appending this list would cause an offset overflow"
        );

        self.elements_builder.extend_from_array(array);
        self.nulls.append_non_null();

        self.offsets_builder.append_value(
            O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
        );
        self.sizes_builder.append_value(
            S::from_usize(num_elements).vortex_expect("Failed to convert from usize to `S`"),
        );

        Ok(())
    }

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

        // We must assert this even in release mode to ensure that the safety comment in
        // `finish_into_listview` is correct.
        assert!(
            ((curr_offset + num_elements) as u64) < O::max_value_as_u64(),
            "appending this list would cause an offset overflow"
        );

        for scalar in elements {
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

        // SAFETY:
        // - Both the offsets and the sizes are non-nullable.
        // - The offsets, sizes, and validity have the same length since we always appended the same
        //   amount.
        // - We checked on construction that the sizes type fits into the offsets.
        // - In every method that adds values to this builder (`append_value`, `append_scalar`, and
        //   `extend_from_array_unchecked`), we checked that `offset + size` does not overflow.
        // - We constructed everything in a way that builds the `ListViewArray` similar to the shape
        //   of a `ListArray`, so we know the resulting array is zero-copyable to a `ListArray`.
        unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, validity)
                .with_zero_copy_to_list(true)
        }
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
            "ListViewBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        let list_scalar = scalar.as_list();
        self.append_value(list_scalar)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        let listview = array.to_listview();
        if listview.is_empty() {
            return;
        }

        // If we do not have the guarantee that the array is zero-copyable to a list, then we have
        // to manually append each scalar.
        if !listview.is_zero_copy_to_list() {
            for i in 0..listview.len() {
                let list = listview
                    .scalar_at(i)
                    .vortex_expect("scalar_at failed in extend_from_array_unchecked");

                self.append_scalar(&list)
                    .vortex_expect("was unable to extend the `ListViewBuilder`")
            }

            return;
        }

        // Otherwise, after removing any leading and trailing elements, we can simply bulk append
        // the entire array.
        let listview = listview
            .rebuild(ListViewRebuildMode::MakeExact)
            .vortex_expect("ListViewArray::rebuild(MakeExact) failed in extend_from_array");
        debug_assert!(listview.is_zero_copy_to_list());

        self.nulls.append_validity_mask(
            array
                .validity_mask()
                .vortex_expect("validity_mask in extend_from_array_unchecked"),
        );

        // Bulk append the new elements (which should have no gaps or overlaps).
        let old_elements_len = self.elements_builder.len();
        self.elements_builder
            .reserve_exact(listview.elements().len());
        self.elements_builder.extend_from_array(listview.elements());
        let new_elements_len = self.elements_builder.len();

        // Reserve enough space for the new views.
        let extend_length = listview.len();
        self.sizes_builder.reserve_exact(extend_length);
        self.offsets_builder.reserve_exact(extend_length);

        // The incoming sizes might have a different type than the builder, so we need to cast.
        let cast_sizes = listview
            .sizes()
            .clone()
            .cast(self.sizes_builder.dtype().clone())
            .vortex_expect(
                "was somehow unable to cast the new sizes to the type of the builder sizes",
            );
        self.sizes_builder.extend_from_array(&cast_sizes);

        // Now we need to adjust all of the offsets by adding the current number of elements in the
        // builder.

        let uninit_range = self.offsets_builder.uninit_range(extend_length);

        // This should be cheap because we didn't compress after rebuilding.
        let new_offsets = listview.offsets().to_primitive();

        match_each_integer_ptype!(new_offsets.ptype(), |A| {
            adjust_and_extend_offsets::<O, A>(
                uninit_range,
                new_offsets,
                old_elements_len,
                new_elements_len,
            );
        })
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.elements_builder.reserve_exact(capacity * 2);
        self.offsets_builder.reserve_exact(capacity);
        self.sizes_builder.reserve_exact(capacity);
        self.nulls.reserve_exact(capacity);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_listview().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::List(self.finish_into_listview())
    }
}

/// Given new offsets, adds them to the `UninitRange` after adding the `old_elements_len` to each
/// offset.
fn adjust_and_extend_offsets<'a, O: IntegerPType, A: IntegerPType>(
    mut uninit_range: UninitRange<'a, O>,
    new_offsets: PrimitiveArray,
    old_elements_len: usize,
    new_elements_len: usize,
) {
    let new_offsets_slice = new_offsets.as_slice::<A>();
    let old_elements_len = O::from_usize(old_elements_len)
        .vortex_expect("the old elements length did not fit into the offset type (impossible)");
    let new_elements_len = O::from_usize(new_elements_len)
        .vortex_expect("the current elements length did not fit into the offset type (impossible)");

    for i in 0..uninit_range.len() {
        let new_offset = O::from_usize(
            new_offsets_slice[i]
                .to_usize()
                .vortex_expect("Offsets must always fit in usize"),
        )
        .vortex_expect("New offset somehow did not fit into the builder's offset type");

        // We have to check this even in release mode to ensure the final `new_unchecked`
        // construction in `finish_into_listview` is valid.
        let adjusted_new_offset = new_offset + old_elements_len;
        assert!(
            adjusted_new_offset <= new_elements_len,
            "[{i}/{}]: {new_offset} + {old_elements_len} \
                = {adjusted_new_offset} <= {new_elements_len} failed",
            uninit_range.len()
        );

        uninit_range.set_value(i, adjusted_new_offset);
    }

    // SAFETY: We have set all the values in the range, and since `offsets` are non-nullable, we are
    // done.
    unsafe { uninit_range.finish() };
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::ListViewBuilder;
    use crate::IntoArray;
    use crate::arrays::ListArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::listview::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;

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
        assert_arrays_eq!(
            listview.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2, 3])
        );

        // Check empty list.
        assert_eq!(listview.list_elements_at(1).unwrap().len(), 0);

        // Check null list.
        assert!(!listview.validity().is_valid(2).unwrap());

        // Check last list: [4, 5].
        assert_arrays_eq!(
            listview.list_elements_at(3).unwrap(),
            PrimitiveArray::from_iter([4i32, 5])
        );
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
        assert_arrays_eq!(
            listview.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([1i32, 2])
        );

        // Verify second list: [3, 4, 5].
        assert_arrays_eq!(
            listview.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([3i32, 4, 5])
        );

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
            assert_arrays_eq!(
                listview2.list_elements_at(i as usize).unwrap(),
                PrimitiveArray::from_iter([i * 10])
            );
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
        assert_eq!(listview.list_elements_at(0).unwrap().len(), 0);
        assert_eq!(listview.list_elements_at(1).unwrap().len(), 0);

        // Next two are nulls.
        assert!(!listview.validity().is_valid(2).unwrap());
        assert!(!listview.validity().is_valid(3).unwrap());

        // Last is the regular list: [10, 20].
        assert_arrays_eq!(
            listview.list_elements_at(4).unwrap(),
            PrimitiveArray::from_iter([10i32, 20])
        );
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
        builder.extend_from_array(&source.into_array());

        // Extend from empty array (should be no-op).
        let empty_source = ListArray::from_iter_opt_slow::<u32, _, Vec<i32>>(
            std::iter::empty::<Option<Vec<i32>>>(),
            Arc::new(I32.into()),
        )
        .unwrap();
        builder.extend_from_array(&empty_source.into_array());

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 4);

        // Check the extended data.
        // First list: [0] (initial data).
        assert_arrays_eq!(
            listview.list_elements_at(0).unwrap(),
            PrimitiveArray::from_iter([0i32])
        );

        // Second list: [1, 2, 3] (from source).
        assert_arrays_eq!(
            listview.list_elements_at(1).unwrap(),
            PrimitiveArray::from_iter([1i32, 2, 3])
        );

        // Third list: null (from source).
        assert!(!listview.validity().is_valid(2).unwrap());

        // Fourth list: [4, 5] (from source).
        assert_arrays_eq!(
            listview.list_elements_at(3).unwrap(),
            PrimitiveArray::from_iter([4i32, 5])
        );
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
    fn test_append_array_as_list() {
        use vortex_buffer::buffer;

        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            ListViewBuilder::<u32, u32>::with_capacity(dtype.clone(), NonNullable, 20, 10);

        // Append a primitive array as a single list entry.
        let arr1 = buffer![1i32, 2, 3].into_array();
        builder.append_array_as_list(&arr1).unwrap();

        // Interleave with a list scalar.
        builder
            .append_value(
                Scalar::list(dtype.clone(), vec![10i32.into(), 11i32.into()], NonNullable)
                    .as_list(),
            )
            .unwrap();

        // Append another primitive array as a single list entry.
        let arr2 = buffer![4i32, 5].into_array();
        builder.append_array_as_list(&arr2).unwrap();

        // Append an empty array as a single list entry (empty list).
        let arr3 = buffer![0i32; 0].into_array();
        builder.append_array_as_list(&arr3).unwrap();

        // Interleave with another list scalar.
        builder
            .append_value(Scalar::list_empty(dtype.clone(), NonNullable).as_list())
            .unwrap();

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), 5);

        // Verify elements array: [1, 2, 3, 10, 11, 4, 5].
        assert_arrays_eq!(
            listview.elements(),
            PrimitiveArray::from_iter([1i32, 2, 3, 10, 11, 4, 5])
        );

        // Verify offsets array.
        assert_arrays_eq!(
            listview.offsets(),
            PrimitiveArray::from_iter([0u32, 3, 5, 7, 7])
        );

        // Verify sizes array.
        assert_arrays_eq!(
            listview.sizes(),
            PrimitiveArray::from_iter([3u32, 2, 2, 0, 0])
        );

        // Test dtype mismatch error.
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity(dtype, NonNullable, 20, 10);
        let wrong_dtype_arr = buffer![1i64, 2, 3].into_array();
        assert!(builder.append_array_as_list(&wrong_dtype_arr).is_err());
    }
}
