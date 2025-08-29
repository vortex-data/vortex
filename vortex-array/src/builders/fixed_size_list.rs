// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::ListScalar;

use crate::arrays::FixedSizeListArray;
use crate::builders::{
    ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder, builder_with_capacity,
};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// The builder for building a [`FixedSizeListArray`].
pub struct FixedSizeListBuilder {
    /// The [`DType`] of the `FixedSizeList`. This **must** be a [`DType::FixedSizeList`].
    dtype: DType,

    /// The builder for the underlying elements of the [`FixedSizeListArray`].
    ///
    /// This builder will have a capacity equal to the `list_size * capacity`.
    elements_builder: Box<dyn ArrayBuilder>,

    /// The null map builder of the [`FixedSizeListArray`].
    ///
    /// We also use this type to store the length of the final output array.
    nulls: LazyNullBufferBuilder,
}

impl FixedSizeListBuilder {
    /// Creates a new `FixedSizeListBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(element_dtype: Arc<DType>, list_size: u32, nullability: Nullability) -> Self {
        Self::with_capacity(
            element_dtype,
            list_size,
            nullability,
            DEFAULT_BUILDER_CAPACITY,
        )
    }

    /// Creates a new `FixedSizeListBuilder` with the given `capacity`.
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        list_size: u32,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let elements_capacity = capacity * list_size as usize;

        let elements_builder = builder_with_capacity(&element_dtype, elements_capacity);
        let fsl_dtype = DType::FixedSizeList(element_dtype, list_size, nullability);
        let nulls = LazyNullBufferBuilder::new(capacity);

        Self {
            dtype: fsl_dtype,
            elements_builder,
            nulls,
        }
    }

    /// Appends a fixed-size list `value` to the builder.
    ///
    /// Note that a [`ListScalar`] can represent both a [`ListArray`] scalar **and** a
    /// [`FixedSizeListArray`] scalar (since a single list cannot know the size of other lists in
    /// fixed-size list arrays without accompanying metadata).
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        if value.len() != self.list_size() as usize {
            vortex_bail!(
                "Tried to append a `ListScalar` with length {} to a `FixedSizeListScalar` \
                    with fixed size of {}",
                value.len(),
                self.list_size()
            );
        }

        let Some(elements) = value.elements() else {
            // If `elements` is `None`, then the `value` is a null value.
            self.append_null();
            return Ok(());
        };

        for scalar in elements {
            // TODO(connor): This is slow, we should be able to append multiple values at once, or
            // the list scalar should hold an Array
            self.elements_builder.append_scalar(&scalar)?;
        }
        self.nulls.append_non_null();

        Ok(())
    }

    /// Finishes the builder directly into a [`FixedSizeListArray`].
    pub fn finish_into_fixed_size_list(&mut self) -> FixedSizeListArray {
        let final_len = self.len();
        assert_eq!(
            self.elements_builder.len(),
            final_len * self.list_size() as usize,
            "elements length must be equal to the array length times the list size"
        );

        // TODO(connor): Use `new_unchecked` here.
        FixedSizeListArray::try_new(
            self.elements_builder.finish(),
            self.list_size(),
            self.nulls.finish_with_nullability(self.dtype.nullability()),
            final_len,
        )
        .vortex_expect("tried to create an invalid `FixedSizeListArray` from a builder")
    }

    /// The [`DType`] of the inner elements. Note that this is **not** the same as the [`DType`] of
    /// the outer `FixedSizeList`.
    pub fn element_dtype(&self) -> &DType {
        let DType::FixedSizeList(element_dtype, ..) = &self.dtype else {
            vortex_panic!(
                "`FixedSizeListBuilder` has an incorrect dtype: {}",
                self.dtype
            );
        };

        element_dtype
    }

    /// The size of each fixed-size list.
    pub fn list_size(&self) -> u32 {
        let DType::FixedSizeList(_, list_size, _) = self.dtype else {
            vortex_panic!(
                "`FixedSizeListBuilder` has an incorrect dtype: {}",
                self.dtype
            );
        };

        list_size
    }

    /// Returns the current length of underlying `elements` array.
    fn elements_len(&self) -> usize {
        self.len() * self.list_size() as usize
    }
}

impl ArrayBuilder for FixedSizeListBuilder {
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

    /// We define the "zero" value of a fixed-size list of size `m` to be a list of `m` zero values
    /// (where "zero value" is defined by the element [`DType`]).
    ///
    /// We append `n * m` of these values to the underlying `elements` array.
    fn append_zeros(&mut self, n: usize) {
        self.elements_builder
            .append_zeros(n * self.list_size() as usize);
        self.nulls.append_n_non_nulls(n);
    }

    /// We define the null value of a fixed-size list of size `m` to be a list of `m` placeholder values.
    ///
    /// We append `n * m` default values to the underlying `elements` array.
    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        assert!(
            self.dtype.is_nullable(),
            "tried to append {n} nulls to a non-nullable array builder"
        );

        let element_count = n * self.list_size() as usize;

        self.elements_builder.append_defaults(element_count);
        self.nulls.append_n_nulls(n);
    }

    /// This will increase the capacity if extending with this `array` would go past the original
    /// capacity.
    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let fsl = array.to_fixed_size_list();
        if fsl.is_empty() {
            return;
        }

        let new_elements = fsl.elements();

        self.elements_builder
            .ensure_capacity(self.elements_len() + new_elements.len());
        self.elements_builder.extend_from_array(new_elements);

        self.nulls.append_validity_mask(array.validity_mask());
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.elements_builder
            .ensure_capacity(capacity * self.list_size() as usize);
        self.nulls.ensure_capacity(capacity);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_fixed_size_list().into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_dtype::PType::I32;
    use vortex_scalar::Scalar;

    use super::FixedSizeListBuilder;
    use crate::array::Array;
    use crate::arrays::FixedSizeListArray;
    use crate::builders::ArrayBuilder;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{IntoArray as _, ToCanonical};

    #[test]
    fn test_empty() {
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::new(I32.into()), 3, NonNullable, 0);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 0);
    }

    #[test]
    fn test_values() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 3, NonNullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype.clone(),
                    vec![1i32.into(), 2i32.into(), 3i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype,
                    vec![4i32.into(), 5i32.into(), 6i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 2);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 6);
        assert_eq!(fsl_array.list_size(), 3);
    }

    #[test]
    fn test_degenerate_size_zero_non_nullable() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            FixedSizeListBuilder::with_capacity(dtype.clone(), 0, NonNullable, 10000000);

        // Append multiple "empty" lists.
        for _ in 0..100 {
            builder
                .append_value(Scalar::fixed_size_list(dtype.clone(), vec![], NonNullable).as_list())
                .unwrap();
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 100);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        // The elements array should be empty since list_size is 0.
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_degenerate_size_zero_nullable() {
        // Use nullable elements since we'll be appending nulls
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 0, Nullable, 10000000);

        // Mix of null and non-null empty lists.
        for i in 0..100 {
            if i % 2 == 0 {
                builder
                    .append_value(
                        Scalar::fixed_size_list(dtype.clone(), vec![], Nullable).as_list(),
                    )
                    .unwrap();
            } else {
                builder.append_null();
            }
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 100);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_capacity_growth() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        // Start with capacity 0.
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 2, NonNullable, 0);

        // Add more items than initial capacity.
        for i in 0..5 {
            builder
                .append_value(
                    Scalar::fixed_size_list(
                        dtype.clone(),
                        vec![(i * 2).into(), (i * 2 + 1).into()],
                        NonNullable,
                    )
                    .as_list(),
                )
                .unwrap();
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 5);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 10);
    }

    #[test]
    fn test_large_size_zero_capacity_empty_result() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        // Large list size but zero capacity and no appends.
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 100000000, NonNullable, 0);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 0);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 100000000);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_nullable_lists_non_nullable_elements() {
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, NonNullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 2, Nullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(dtype.clone(), vec![1i32.into(), 2i32.into()], Nullable)
                    .as_list(),
            )
            .unwrap();

        builder.append_null();

        builder
            .append_value(
                Scalar::fixed_size_list(dtype, vec![3i32.into(), 4i32.into()], Nullable).as_list(),
            )
            .unwrap();

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 3);

        let fsl_array = fsl.to_fixed_size_list();
        assert!(fsl_array.validity().is_valid(0));
        assert!(!fsl_array.validity().is_valid(1));
        assert!(fsl_array.validity().is_valid(2));
    }

    #[test]
    fn test_non_nullable_lists_nullable_elements() {
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 3, NonNullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype.clone(),
                    vec![
                        Scalar::primitive(1i32, Nullable),
                        Scalar::null(dtype.as_ref().clone()),
                        Scalar::primitive(3i32, Nullable),
                    ],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype,
                    vec![
                        Scalar::primitive(4i32, Nullable),
                        Scalar::primitive(5i32, Nullable),
                        Scalar::primitive(6i32, Nullable),
                    ],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 2);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 6);
    }

    #[test]
    fn test_append_zeros() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 3, NonNullable, 0);

        builder.append_zeros(5);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 5);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 3);
        assert_eq!(fsl_array.elements().len(), 15);

        // Check that all elements are zeros.
        let elements_array = fsl_array.elements().to_primitive();
        let elements = elements_array.as_slice::<i32>();
        assert!(elements.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_append_nulls() {
        // Elements must be nullable if we're going to append null lists
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 2, Nullable, 0);

        assert_eq!(builder.dtype().nullability(), Nullable);
        builder.append_nulls(3);
        assert_eq!(builder.len(), 3);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 3);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 2);

        // Check that all lists are null.
        for i in 0..3 {
            assert!(!fsl_array.validity().is_valid(i));
        }
    }

    #[test]
    fn test_append_zeros_degenerate() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 0, NonNullable, 0);

        assert_eq!(builder.len(), 0);
        builder.append_zeros(1000);
        assert_eq!(builder.len(), 1000);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 1000);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_invalid_size_error() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 3, NonNullable, 0);

        // Try to append a list with wrong size.
        let result = builder.append_value(
            Scalar::fixed_size_list(
                dtype,
                vec![1i32.into(), 2i32.into()], // Only 2 elements, not 3.
                NonNullable,
            )
            .as_list(),
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("with fixed size of 3")
        );
    }

    #[test]
    fn test_extend_from_array() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create a source array.
        let source = FixedSizeListArray::new(
            crate::arrays::PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array(),
            2,
            Validity::from_iter([true, false, true]),
            3,
        );

        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 2, Nullable, 0);

        let source_array = source.into_array();
        builder.extend_from_array(&source_array);
        builder.extend_from_array(&source_array);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 6);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 12);

        // Check validity pattern is repeated.
        assert!(fsl_array.validity().is_valid(0));
        assert!(!fsl_array.validity().is_valid(1));
        assert!(fsl_array.validity().is_valid(2));
        assert!(fsl_array.validity().is_valid(3));
        assert!(!fsl_array.validity().is_valid(4));
        assert!(fsl_array.validity().is_valid(5));
    }

    #[test]
    fn test_extend_degenerate_arrays() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create degenerate source arrays (size = 0).
        let source1 = FixedSizeListArray::new(
            crate::arrays::PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            0,
            Validity::from_iter([true, false, true]),
            3,
        );

        let source2 = FixedSizeListArray::new(
            crate::arrays::PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            0,
            Validity::from_iter([false, true]),
            2,
        );

        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 0, Nullable, 0);

        builder.extend_from_array(&source1.into_array());
        builder.extend_from_array(&source2.into_array());

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 5);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);

        // Check validity pattern.
        assert!(fsl_array.validity().is_valid(0));
        assert!(!fsl_array.validity().is_valid(1));
        assert!(fsl_array.validity().is_valid(2));
        assert!(!fsl_array.validity().is_valid(3));
        assert!(fsl_array.validity().is_valid(4));
    }

    #[test]
    fn test_extend_empty_array() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create an empty source array.
        let source = FixedSizeListArray::new(
            crate::arrays::PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            3,
            Validity::NonNullable,
            0,
        );

        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 3, NonNullable, 0);

        // Add some initial data.
        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype,
                    vec![1i32.into(), 2i32.into(), 3i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        // Extend with empty array (should be no-op).
        builder.extend_from_array(&source.into_array());

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 1);
    }

    #[test]
    fn test_mixed_operations() {
        // Use nullable elements since we'll be appending nulls
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype.clone(), 2, Nullable, 0);

        // Mix of operations.
        builder
            .append_value(
                Scalar::fixed_size_list(
                    dtype,
                    vec![
                        Scalar::primitive(1i32, Nullable),
                        Scalar::primitive(2i32, Nullable),
                    ],
                    Nullable,
                )
                .as_list(),
            )
            .unwrap();
        builder.append_null();
        builder.append_zeros(2);
        builder.append_null();

        // Create source with nullable elements to match builder dtype
        let source = FixedSizeListArray::new(
            crate::arrays::PrimitiveArray::from_option_iter([Some(5i32), Some(6)]).into_array(),
            2,
            Validity::AllValid,
            1,
        );
        builder.extend_from_array(&source.into_array());

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 6);

        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 12);

        // Check validity.
        assert!(fsl_array.validity().is_valid(0)); // append_value
        assert!(!fsl_array.validity().is_valid(1)); // append_null
        assert!(fsl_array.validity().is_valid(2)); // append_zeros
        assert!(fsl_array.validity().is_valid(3)); // append_zeros
        assert!(!fsl_array.validity().is_valid(4)); // append_nulls
        assert!(fsl_array.validity().is_valid(5)); // extend_from_array
    }
}
