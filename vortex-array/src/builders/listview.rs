// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ListView Builder Implementation.
//!
//! A builder for [`ListViewArray`] that tracks both offsets and sizes.
//!
//! Unlike [`ListArray`] which only tracks offsets, [`ListViewArray`] stores both offsets and sizes
//! in separate arrays for better compression.

use std::sync::Arc;

use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{ListScalar, Scalar};

use super::lazy_null_builder::LazyNullBufferBuilder;
use crate::array::{Array, ArrayRef, IntoArray};
use crate::arrays::{ListViewArray, list_view_from_list};
use crate::builders::{
    ArrayBuilder, DEFAULT_BUILDER_CAPACITY, PrimitiveBuilder, builder_with_capacity,
};
use crate::{Canonical, OffsetPType, ToCanonical};

/// A builder for creating [`ListViewArray`] instances.
///
/// This builder tracks both offsets and sizes using potentially different integer types for memory
/// efficiency. For example, you might use `u64` for offsets but only `u8` for sizes if your lists
/// are small.
///
/// Any combination of [`OffsetPType`] types are valid, as long as the type of `sizes` can fit into
/// the type of `offsets`.
///
/// # Example
/// ```ignore
/// let mut builder = ListViewBuilder::<u32, u8>::new(&DType::I32, NonNullable);
/// builder.append_value(&[1, 2, 3]);
/// builder.append_value(&[4, 5]);
/// let array = builder.finish();
/// ```
pub struct ListViewBuilder<O: OffsetPType, S: OffsetPType> {
    dtype: DType,
    elements_builder: Box<dyn ArrayBuilder>,
    offsets_builder: PrimitiveBuilder<O>,
    sizes_builder: PrimitiveBuilder<S>,
    nulls: LazyNullBufferBuilder,
}

impl<O: OffsetPType, S: OffsetPType> ListViewBuilder<O, S> {
    /// Creates a new `ListViewBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::with_capacity(element_dtype, nullability, DEFAULT_BUILDER_CAPACITY)
    }

    /// Create a new builder with a custom value builder.
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        // We arbitrarily choose 2 times the number of list scalars for the capacity of the elements
        // builder since we cannot know this ahead of time.
        let elements_capacity = capacity * 2;
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

    /// Append a list of values to the builder.
    ///
    /// This method extends the value builder with the provided values and records
    /// the offset and size of the new list.
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        let Some(elements) = value.elements() else {
            // If `elements` is `None`, then the `value` is a null value.
            self.append_null();
            return Ok(());
        };

        let num_elements = elements.len();

        for scalar in elements {
            // TODO(connor): This is slow, we should be able to append multiple values at once, or
            // the list scalar should hold an Array
            self.elements_builder.append_scalar(&scalar)?;
        }
        self.nulls.append_non_null();

        self.offsets_builder.append_value(
            O::from_usize(self.elements_builder.len())
                .vortex_expect("Failed to convert from usize to `O`"),
        );

        self.sizes_builder.append_value(
            S::from_usize(num_elements).vortex_expect("Failed to convert from usize to `S`"),
        );

        Ok(())
    }

    /// Finishes the builder directly into a [`ListViewArray`].
    fn finish_into_listview(&mut self) -> ListViewArray {
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
        let DType::FixedSizeList(element_dtype, ..) = &self.dtype else {
            vortex_panic!("`ListViewBuilder` has an incorrect dtype: {}", self.dtype);
        };

        element_dtype
    }
}

impl<O: OffsetPType, S: OffsetPType> ArrayBuilder for ListViewBuilder<O, S> {
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
        let curr_len = self.offsets_builder.len();
        debug_assert_eq!(curr_len, self.sizes_builder.len());
        debug_assert_eq!(curr_len, self.nulls.len());

        // Since we consider the "zero" element of a list an empty list, we simply update the
        // `offsets` and `sizes` metadata to add an empty list.
        for _ in 0..n {
            self.offsets_builder.append_value(
                O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            );
            self.sizes_builder.append_value(S::zero());
        }

        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        let curr_len = self.offsets_builder.len();
        debug_assert_eq!(curr_len, self.sizes_builder.len());
        debug_assert_eq!(curr_len, self.nulls.len());

        // A null list can have any representation, but we choose to use the zero representation.
        for _ in 0..n {
            self.offsets_builder.append_value(
                O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
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

    fn set_validity(&mut self, validity: Mask) {
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
