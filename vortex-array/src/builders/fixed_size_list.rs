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
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::builders::builder_with_capacity;
use crate::canonical::Canonical;
#[expect(deprecated)]
use crate::canonical::ToCanonical as _;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

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
    nulls: LazyBitBufferBuilder,
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
        let nulls = LazyBitBufferBuilder::new(capacity);

        Self {
            dtype: fsl_dtype,
            elements_builder,
            nulls,
        }
    }

    /// Appends an array as a single non-null list entry to the builder.
    ///
    /// The input `array` must have the same dtype as the element dtype of this list builder, and
    /// its length must match the fixed list size.
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
        vortex_ensure!(
            array.len() == self.list_size() as usize,
            "Array length {} does not match fixed list size {}",
            array.len(),
            self.list_size()
        );

        self.elements_builder.extend_from_array(array);
        self.nulls.append_non_null();

        Ok(())
    }

    /// Appends a fixed-size list `value` to the builder.
    ///
    /// Note that a [`ListScalar`] can represent both a [`ListArray`] scalar **and** a
    /// [`FixedSizeListArray`] scalar (since a single list cannot know the size of other lists in
    /// fixed-size list arrays without accompanying metadata).
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        let Some(elements) = value.elements() else {
            // If `elements` is `None`, then the `value` is a null value.
            self.append_null();
            return Ok(());
        };

        if value.len() != self.list_size() as usize {
            vortex_bail!(
                "Tried to append a `ListScalar` with length {} to a `FixedSizeListScalar` \
                    with fixed size of {}",
                value.len(),
                self.list_size()
            );
        }

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

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "FixedSizeListBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        let list_scalar = scalar.as_list();
        self.append_value(list_scalar)
    }

    /// This will increase the capacity if extending with this `array` would go past the original
    /// capacity.
    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        #[expect(deprecated)]
        let fsl = array.to_fixed_size_list();
        if fsl.is_empty() {
            return;
        }

        self.elements_builder.extend_from_array(fsl.elements());
        self.nulls.append_validity_mask(
            array
                .validity()
                .vortex_expect("validity_mask in extend_from_array_unchecked")
                .execute_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("Failed to compute validity mask"),
        );
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.elements_builder
            .reserve_exact(additional * self.list_size() as usize);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_fixed_size_list().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::FixedSizeList(self.finish_into_fixed_size_list())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::FixedSizeListBuilder;
    use crate::IntoArray as _;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
    use crate::builders::ArrayBuilder;
    use crate::builders::fixed_size_list::FixedSizeListArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

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
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(
                    Arc::clone(&dtype),
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 6);
        assert_eq!(fsl_array.list_size(), 3);
    }

    #[test]
    fn test_degenerate_size_zero_non_nullable() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 0, NonNullable, 10000000);

        // Append multiple "empty" lists.
        for _ in 0..100 {
            builder
                .append_value(
                    Scalar::fixed_size_list(Arc::clone(&dtype), vec![], NonNullable).as_list(),
                )
                .unwrap();
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 100);

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        // The elements array should be empty since list_size is 0.
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_degenerate_size_zero_nullable() {
        // Use nullable elements since we'll be appending nulls
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 0, Nullable, 10000000);

        // Mix of null and non-null empty lists.
        for i in 0..100 {
            if i % 2 == 0 {
                builder
                    .append_value(
                        Scalar::fixed_size_list(Arc::clone(&dtype), vec![], Nullable).as_list(),
                    )
                    .unwrap();
            } else {
                builder.append_null();
            }
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 100);

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_capacity_growth() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        // Start with capacity 0.
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 2, NonNullable, 0);

        // Add more items than initial capacity.
        for i in 0..5 {
            builder
                .append_value(
                    Scalar::fixed_size_list(
                        Arc::clone(&dtype),
                        vec![(i * 2).into(), (i * 2 + 1).into()],
                        NonNullable,
                    )
                    .as_list(),
                )
                .unwrap();
        }

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 5);

        #[expect(deprecated)]
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 100000000);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_nullable_lists_non_nullable_elements() {
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, NonNullable));
        let mut builder = FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 2, Nullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(
                    Arc::clone(&dtype),
                    vec![1i32.into(), 2i32.into()],
                    Nullable,
                )
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(2)
                .unwrap()
        );
    }

    #[test]
    fn test_non_nullable_lists_nullable_elements() {
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 0);

        builder
            .append_value(
                Scalar::fixed_size_list(
                    Arc::clone(&dtype),
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

        #[expect(deprecated)]
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 3);
        assert_eq!(fsl_array.elements().len(), 15);

        // Check that all elements are zeros.
        #[expect(deprecated)]
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 2);

        // Check that all lists are null.
        for i in 0..3 {
            assert!(
                !fsl_array
                    .validity()
                    .vortex_expect("fixed-size-list validity should be derivable")
                    .is_valid(i)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_append_scalar_nulls() {
        // Elements must be nullable if we're going to append null lists
        let dtype: Arc<DType> = Arc::new(DType::Primitive(I32, Nullable));
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 2, Nullable, 0);

        assert_eq!(builder.dtype().nullability(), Nullable);
        builder
            .append_scalar(&Scalar::null(builder.dtype().clone()))
            .unwrap();
        assert_eq!(builder.len(), 1);

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 1);

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 2);

        // Check that all lists are null.
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);
    }

    #[test]
    fn test_invalid_size_error() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 0);

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
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
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

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 12);

        // Check validity pattern is repeated.
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(2)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(3)
                .unwrap()
        );
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(4)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(5)
                .unwrap()
        );
    }

    #[test]
    fn test_extend_degenerate_arrays() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create degenerate source arrays (size = 0).
        let source1 = FixedSizeListArray::new(
            PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            0,
            Validity::from_iter([true, false, true]),
            3,
        );

        let source2 = FixedSizeListArray::new(
            PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            0,
            Validity::from_iter([false, true]),
            2,
        );

        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 0, Nullable, 0);

        builder.extend_from_array(&source1.into_array());
        builder.extend_from_array(&source2.into_array());

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 5);

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.list_size(), 0);
        assert_eq!(fsl_array.elements().len(), 0);

        // Check validity pattern.
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(2)
                .unwrap()
        );
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(3)
                .unwrap()
        );
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(4)
                .unwrap()
        );
    }

    #[test]
    fn test_extend_empty_array() {
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Create an empty source array.
        let source = FixedSizeListArray::new(
            PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
            3,
            Validity::NonNullable,
            0,
        );

        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 0);

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
        let mut builder = FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 2, Nullable, 0);

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
            PrimitiveArray::from_option_iter([Some(5i32), Some(6)]).into_array(),
            2,
            Validity::AllValid,
            1,
        );
        builder.extend_from_array(&source.into_array());

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 6);

        #[expect(deprecated)]
        let fsl_array = fsl.to_fixed_size_list();
        assert_eq!(fsl_array.elements().len(), 12);

        // Check validity.
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        ); // append_value
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(1)
                .unwrap()
        ); // append_null
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(2)
                .unwrap()
        ); // append_zeros
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(3)
                .unwrap()
        ); // append_zeros
        assert!(
            !fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(4)
                .unwrap()
        ); // append_nulls
        assert!(
            fsl_array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(5)
                .unwrap()
        ); // extend_from_array
    }

    #[test]
    fn test_append_scalar() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 2, Nullable, 10);

        // Test appending a valid fixed-size list.
        let list_scalar1 =
            Scalar::fixed_size_list(Arc::clone(&dtype), vec![1i32.into(), 2i32.into()], Nullable);
        builder.append_scalar(&list_scalar1).unwrap();

        // Test appending another list.
        let list_scalar2 =
            Scalar::fixed_size_list(Arc::clone(&dtype), vec![3i32.into(), 4i32.into()], Nullable);
        builder.append_scalar(&list_scalar2).unwrap();

        // Test appending null via builder method (since fixed-size list null handling is special).
        builder.append_null();

        let array = builder.finish_into_fixed_size_list();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.

        let scalar0 = array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let list0 = scalar0.as_list();
        assert_eq!(list0.len(), 2);
        if let Some(list0_items) = list0.elements() {
            assert_eq!(list0_items[0].as_primitive().typed_value::<i32>(), Some(1));
            assert_eq!(list0_items[1].as_primitive().typed_value::<i32>(), Some(2));
        }

        let scalar1 = array
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let list1 = scalar1.as_list();
        assert_eq!(list1.len(), 2);
        if let Some(list1_items) = list1.elements() {
            assert_eq!(list1_items[0].as_primitive().typed_value::<i32>(), Some(3));
            assert_eq!(list1_items[1].as_primitive().typed_value::<i32>(), Some(4));
        }

        // Check validity - first two should be valid, third should be null.
        assert!(
            array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            !array
                .validity()
                .vortex_expect("fixed-size-list validity should be derivable")
                .is_valid(2)
                .unwrap()
        );

        // Test wrong dtype error.
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 2, NonNullable, 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }

    #[test]
    fn test_append_array_as_list() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 10);

        // Append a primitive array as a single list entry.
        let arr1 = buffer![1i32, 2, 3].into_array();
        builder.append_array_as_list(&arr1).unwrap();

        // Interleave with a list scalar.
        builder
            .append_value(
                Scalar::fixed_size_list(
                    Arc::clone(&dtype),
                    vec![10i32.into(), 11i32.into(), 12i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        // Append another primitive array as a single list entry.
        let arr2 = buffer![4i32, 5, 6].into_array();
        builder.append_array_as_list(&arr2).unwrap();

        // Interleave with another list scalar.
        builder
            .append_value(
                Scalar::fixed_size_list(
                    Arc::clone(&dtype),
                    vec![20i32.into(), 21i32.into(), 22i32.into()],
                    NonNullable,
                )
                .as_list(),
            )
            .unwrap();

        let fsl = builder.finish_into_fixed_size_list();
        assert_eq!(fsl.len(), 4);
        assert_eq!(fsl.list_size(), 3);

        // Verify elements array: [1, 2, 3, 10, 11, 12, 4, 5, 6, 20, 21, 22].
        #[expect(deprecated)]
        let elements = fsl.elements().to_primitive();
        assert_eq!(
            elements.as_slice::<i32>(),
            &[1, 2, 3, 10, 11, 12, 4, 5, 6, 20, 21, 22]
        );

        // Test dtype mismatch error.
        let mut builder =
            FixedSizeListBuilder::with_capacity(Arc::clone(&dtype), 3, NonNullable, 10);
        let wrong_dtype_arr = buffer![1i64, 2, 3].into_array();
        assert!(builder.append_array_as_list(&wrong_dtype_arr).is_err());

        // Test length mismatch error.
        let mut builder = FixedSizeListBuilder::with_capacity(dtype, 3, NonNullable, 10);
        let wrong_len_arr = buffer![1i32, 2].into_array();
        assert!(builder.append_array_as_list(&wrong_len_arr).is_err());
    }
}
