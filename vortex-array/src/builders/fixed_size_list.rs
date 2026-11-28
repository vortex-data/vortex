// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::RepeatedArray;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::builders::ArrayBuilder;
use crate::builders::ChildBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::ValidityBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

/// The builder for building a [`FixedSizeListArray`].
pub struct FixedSizeListBuilder {
    /// The [`DType`] of the `FixedSizeList`. This **must** be a [`DType::FixedSizeList`].
    dtype: DType,

    /// The builder for the underlying elements of the [`FixedSizeListArray`].
    ///
    /// This builder will have a capacity equal to the `list_size * capacity`.
    elements_builder: ChildBuilder,

    /// The null map builder of the [`FixedSizeListArray`].
    ///
    /// We also use this type to store the length of the final output array.
    nulls: ValidityBuilder,
}

impl FixedSizeListBuilder {
    /// Creates a new `FixedSizeListBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(element_dtype: Arc<DType>, list_size: u32, nullability: Nullability) -> Self {
        Self::new_in(
            element_dtype,
            list_size,
            nullability,
            BufferAllocatorRef::static_ref(),
        )
    }

    /// Creates a new `FixedSizeListBuilder` with the default capacity using `allocator`.
    pub fn new_in(
        element_dtype: Arc<DType>,
        list_size: u32,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_in(
            element_dtype,
            list_size,
            nullability,
            DEFAULT_BUILDER_CAPACITY,
            allocator,
        )
    }

    /// Creates a new `FixedSizeListBuilder` with the given `capacity`.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        list_size: u32,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        Self::with_capacity_in(
            element_dtype,
            list_size,
            nullability,
            capacity,
            BufferAllocatorRef::static_ref(),
        )
    }

    /// Creates a new `FixedSizeListBuilder` with `capacity` using `allocator`.
    pub fn with_capacity_in(
        element_dtype: Arc<DType>,
        list_size: u32,
        nullability: Nullability,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        let elements_capacity = capacity * list_size as usize;

        let elements_builder =
            ChildBuilder::with_capacity(&element_dtype, elements_capacity, allocator);
        let fsl_dtype = DType::FixedSizeList(element_dtype, list_size, nullability);
        let nulls = ValidityBuilder::new(capacity, allocator);

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
    pub fn append_array_as_list(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(array.dtype(), self.element_dtype());
        vortex_ensure_eq!(array.len(), self.list_size() as usize);

        self.elements_builder.append_array(array, ctx)?;
        self.nulls.append_non_null();

        Ok(())
    }

    /// Appends `array` as `n` identical non-null lists.
    ///
    /// A fixed-size list array holds its elements back to back, so `n` identical lists are the
    /// array's elements tiled `n` times - there is no layout that lets the rows share one range of
    /// elements the way a list view's can. The tiling costs nothing to build even so: the elements
    /// go in as a [`RepeatedArray`] of `array` repeated `n` times, so the tile's values are stored
    /// once however many rows reference them, and the child holds the whole run as one chunk.
    ///
    /// A caller with a run of appends to make should hand over the same `array` each time rather
    /// than rebuild it, which is the whole reason this takes an array instead of a scalar.
    pub fn append_array_as_repeated_list(
        &mut self,
        array: &ArrayRef,
        n: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(array.dtype(), self.element_dtype());
        vortex_ensure_eq!(array.len(), self.list_size() as usize);

        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            return self.append_array_as_list(array, ctx);
        }

        let len = n.checked_mul(array.len()).ok_or_else(|| {
            vortex_err!(
                "{n} repeated lists of {} elements overflow usize",
                array.len()
            )
        })?;
        // Optimizing collapses a constant tile into one constant run.
        let repeated = RepeatedArray::try_new(array.clone(), len)?
            .into_array()
            .optimize_ctx(&*ctx)?;
        self.elements_builder.append_array(&repeated, ctx)?;
        self.nulls.append_n_non_nulls(n);

        Ok(())
    }

    /// Appends the values of a canonical [`FixedSizeListArray`] to the builder, recursing into the
    /// elements builder.
    pub(crate) fn append_fixed_size_list_array(
        &mut self,
        array: ArrayView<'_, FixedSizeList>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }

        self.elements_builder.append_array(array.elements(), ctx)?;
        self.nulls.append_validity(array.validity()?, array.len());
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

        // SAFETY: The assertion above checks that the elements length is `final_len * list_size`,
        // and the validity is built from `self.nulls`, whose length is `final_len`.
        unsafe {
            FixedSizeListArray::new_unchecked(
                self.elements_builder.finish(),
                self.list_size(),
                self.nulls.finish_with_nullability(self.dtype.nullability()),
                final_len,
            )
        }
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
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "FixedSizeListBuilder received a scalar with the wrong dtype"
        );

        let list_scalar = scalar.as_list();
        self.append_value(list_scalar)
    }

    /// This will increase the capacity if extending with this `array` would go past the original
    /// capacity.
    fn reserve_chunks(&mut self, additional: usize) {
        self.elements_builder.reserve_chunks(additional);
        self.nulls.reserve_runs(additional);
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.elements_builder
            .reserve_exact(additional * self.list_size() as usize);
        self.nulls.reserve_exact(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_fixed_size_list().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::FixedSizeList(self.finish_into_fixed_size_list())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::BufferAllocatorRef;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::FixedSizeListBuilder;
    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
    use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_large_size_zero_capacity_empty_result() -> VortexResult<()> {
        let dtype: Arc<DType> = Arc::new(I32.into());
        // Large list size with zero capacity and no appends must not allocate elements.
        let mut builder = FixedSizeListBuilder::with_capacity_in(
            dtype,
            100_000_000,
            NonNullable,
            0,
            BufferAllocatorRef::static_ref(),
        );

        let fsl = builder.finish();
        assert_eq!(fsl.len(), 0);

        let mut ctx = SESSION.create_execution_ctx();
        let fsl_array = fsl.execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(fsl_array.list_size(), 100_000_000);
        assert_eq!(fsl_array.elements().len(), 0);
        Ok(())
    }

    #[test]
    fn test_invalid_size_error() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = FixedSizeListBuilder::with_capacity_in(
            Arc::clone(&dtype),
            3,
            NonNullable,
            0,
            BufferAllocatorRef::static_ref(),
        );

        let err = builder
            .append_value(
                Scalar::fixed_size_list(dtype, vec![1i32.into(), 2i32.into()], NonNullable)
                    .as_list(),
            )
            .expect_err("wrong fixed size must fail");
        assert!(err.to_string().contains("with fixed size of 3"));
    }

    #[test]
    fn test_append_array_as_list() -> VortexResult<()> {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = FixedSizeListBuilder::with_capacity_in(
            Arc::clone(&dtype),
            3,
            NonNullable,
            4,
            BufferAllocatorRef::static_ref(),
        );

        builder.append_array_as_list(&buffer![1i32, 2, 3].into_array(), &mut ctx)?;
        builder.append_value(
            Scalar::fixed_size_list(
                Arc::clone(&dtype),
                vec![10i32.into(), 11i32.into(), 12i32.into()],
                NonNullable,
            )
            .as_list(),
        )?;
        builder.append_array_as_list(&buffer![4i32, 5, 6].into_array(), &mut ctx)?;

        let fsl = builder.finish_into_fixed_size_list();
        assert_eq!(fsl.len(), 3);
        assert_arrays_eq!(
            fsl.elements(),
            PrimitiveArray::from_iter([1i32, 2, 3, 10, 11, 12, 4, 5, 6]),
            &mut ctx
        );

        // Wrong element dtype and wrong list length must both error.
        let mut err_builder = FixedSizeListBuilder::with_capacity_in(
            dtype,
            3,
            NonNullable,
            4,
            BufferAllocatorRef::static_ref(),
        );
        assert!(
            err_builder
                .append_array_as_list(&buffer![1i64, 2, 3].into_array(), &mut ctx)
                .is_err()
        );
        assert!(
            err_builder
                .append_array_as_list(&buffer![1i32, 2].into_array(), &mut ctx)
                .is_err()
        );
        Ok(())
    }
}
