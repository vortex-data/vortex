// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builders::ArrayBuilder;
use crate::builders::ChildBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::PrimitiveBuilder;
use crate::builders::ValidityBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::OffsetBuilderPType;
use crate::match_each_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

/// The builder for building a [`ListArray`], parametrized by the [`OffsetBuilderPType`] of the
/// `offsets` builder.
pub struct ListBuilder<O: OffsetBuilderPType> {
    /// The [`DType`] of the [`ListArray`]. This **must** be a [`DType::List`].
    dtype: DType,

    /// The builder for the underlying elements of the [`ListArray`].
    elements_builder: ChildBuilder,

    /// The builder for the `offsets` into the `elements` array.
    offsets_builder: PrimitiveBuilder<O>,

    /// The null map builder of the [`ListArray`].
    nulls: ValidityBuilder,
}

impl<O: OffsetBuilderPType> ListBuilder<O> {
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
        let elements_builder = ChildBuilder::with_capacity(value_dtype.as_ref(), elements_capacity);
        let mut offsets_builder = PrimitiveBuilder::<O>::with_capacity(NonNullable, capacity + 1);

        // The first offset is always 0 and represents an empty list.
        offsets_builder.append_zero();

        Self {
            elements_builder,
            offsets_builder,
            nulls: ValidityBuilder::new(capacity),
            dtype: DType::List(value_dtype, nullability),
        }
    }

    /// Appends an array as a single non-null list entry to the builder.
    ///
    /// The input `array` must have the same dtype as the element dtype of this list builder.
    ///
    /// Note that the list entry will be non-null but the elements themselves are allowed to be null
    /// (only if the elements [`DType`] in nullable, of course).
    pub fn append_array_as_list(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == self.element_dtype(),
            "Array dtype {:?} does not match list element dtype {:?}",
            array.dtype(),
            self.element_dtype()
        );

        self.elements_builder.append_array(array, ctx)?;
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

    /// Appends the same list `value` `n` times.
    ///
    /// A `ListArray`'s offsets can only describe contiguous, in-order lists, so the elements go in
    /// once per row - there is nothing to share, unlike
    /// [`ListViewBuilder::append_constant_list`](crate::builders::ListViewBuilder::append_constant_list).
    /// What this does save is the array machinery: canonicalizing the run and appending it back
    /// costs a `ListViewArray` construction, a rebuild and two offset casts that the values
    /// themselves do not need.
    pub fn append_constant_list(
        &mut self,
        value: ListScalar,
        n: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if n == 0 {
            return Ok(());
        }

        let Some(elements) = value.elements() else {
            if self.dtype.nullability() == NonNullable {
                vortex_bail!("Cannot append null value to non-nullable list");
            }
            self.append_nulls(n);
            return Ok(());
        };

        self.offsets_builder.reserve_exact(n);
        for _ in 0..n {
            for scalar in &elements {
                self.elements_builder.append_scalar(scalar)?;
            }
            self.offsets_builder.append_value(
                O::from_usize(self.elements_builder.len())
                    .vortex_expect("Failed to convert from usize to O"),
            );
        }
        self.nulls.append_n_non_nulls(n);

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

    /// Appends the values of a [`List`]-encoded `array` to this builder.
    ///
    /// List encodings dispatch here through
    /// [`match_each_list_builder!`](crate::match_each_list_builder) because the concrete list
    /// builders are generic over their offset integer type, which cannot be named through a
    /// `dyn ArrayBuilder`.
    pub fn append_list_array(
        &mut self,
        array: ArrayView<'_, List>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }

        self.nulls.append_validity(array.validity()?, array.len());

        let num_lists = array.len();
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |OffsetType| {
            let offsets = offsets.as_slice::<OffsetType>();
            let first: usize = offsets[0].as_();
            let last: usize = offsets[num_lists].as_();

            // Lists in a `ListArray` are contiguous, so the referenced elements can be appended
            // in bulk and the offsets rebased onto this builder's elements.
            let elements_base = self.elements_builder.len();
            if last > first {
                self.elements_builder
                    .append_array(&array.elements().slice(first..last)?, ctx)?;
            }

            self.offsets_builder.reserve_exact(num_lists);
            let mut offsets_range = self.offsets_builder.uninit_range(num_lists);
            for i in 0..num_lists {
                let end: usize = offsets[i + 1].as_();
                offsets_range.set_value(
                    i,
                    O::from_usize(end - first + elements_base)
                        .vortex_expect("Failed to convert offset"),
                );
            }
            // SAFETY: We have initialized all `num_lists` values, and since the `offsets` array is
            // non-nullable, we are done.
            unsafe { offsets_range.finish() };
        });
        Ok(())
    }

    /// Appends the values of a [`ListView`]-encoded `array` to this builder.
    ///
    /// See [`append_list_array`](Self::append_list_array); this is the same hook for the canonical
    /// [`ListViewArray`] encoding.
    ///
    /// A `ListArray`'s offsets can only describe contiguous, in-order lists, so views laid out any
    /// other way (overlapping, out of order, or with interior gaps) are flattened first.
    pub fn append_listview_array(
        &mut self,
        array: ArrayView<'_, ListView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }

        self.nulls.append_validity(array.validity()?, array.len());

        // Flatten the views into the only layout `ListArray` offsets can express. This is a cheap
        // clone when they already are laid out that way, and the flattened result keeps the
        // original validity, so the null map appended above still describes it.
        let array = array
            .into_owned()
            .rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)?;
        debug_assert!(array.is_zero_copy_to_list());

        // Note that `ListViewArray` has `n` offsets and sizes, not `n+1` offsets like `ListArray`.
        let elements = array.elements();
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let sizes = array.sizes().clone().execute::<PrimitiveArray>(ctx)?;

        match_each_integer_ptype!(offsets.ptype(), |OffsetType| {
            match_each_integer_ptype!(sizes.ptype(), |SizeType| {
                extend_from_listview(
                    self,
                    elements,
                    offsets.as_slice::<OffsetType>(),
                    sizes.as_slice::<SizeType>(),
                    ctx,
                )?
            })
        });
        Ok(())
    }
}

/// Appends the lists of a zero-copy-to-list [`ListViewArray`] (`n` offsets and sizes) into a
/// [`ListBuilder`], converting into the `ListArray` (`n + 1` offsets) layout.
///
/// The caller must have made `new_offsets` and `new_sizes` zero-copyable to a `ListArray`, so the
/// lists they describe are contiguous and in order — which is the only layout `ListArray` offsets
/// can express. That lets the referenced elements be appended in bulk, with the offsets rebased
/// onto this builder's elements, instead of appending a slice per list.
fn extend_from_listview<O, OffsetType, SizeType>(
    builder: &mut ListBuilder<O>,
    new_elements: &ArrayRef,
    new_offsets: &[OffsetType],
    new_sizes: &[SizeType],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    O: OffsetBuilderPType,
    OffsetType: IntegerPType,
    SizeType: IntegerPType,
{
    let num_lists = new_offsets.len();
    debug_assert_eq!(num_lists, new_sizes.len());

    // Leading and trailing unreferenced elements are allowed even in a zero-copy-to-list layout,
    // so the referenced range is bounded by the first list's start and the last list's end.
    let first: usize = new_offsets[0].as_();
    let last: usize = new_offsets[num_lists - 1].as_() + new_sizes[num_lists - 1].as_();

    let elements_base = builder.elements_builder.len();
    if last > first {
        builder
            .elements_builder
            .append_array(&new_elements.slice(first..last)?, ctx)?;
    }

    builder.offsets_builder.reserve_exact(num_lists);
    let mut offsets_range = builder.offsets_builder.uninit_range(num_lists);
    for i in 0..num_lists {
        let end: usize = new_offsets[i].as_() + new_sizes[i].as_();
        offsets_range.set_value(
            i,
            O::from_usize(end - first + elements_base).vortex_expect("Failed to convert offset"),
        );
    }

    // SAFETY: We have initialized all `num_lists` values, and since the `offsets` array is
    // non-nullable, we are done.
    unsafe { offsets_range.finish() };
    Ok(())
}

impl<O: OffsetBuilderPType> ArrayBuilder for ListBuilder<O> {
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

    fn reserve_exact(&mut self, additional: usize) {
        self.elements_builder.reserve_exact(additional);
        self.offsets_builder.reserve_exact(additional);
        self.nulls.reserve_exact(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_list().into_array()
    }

    fn finish_into_canonical(&mut self, ctx: &mut ExecutionCtx) -> Canonical {
        let listview = self
            .finish()
            .execute::<ListViewArray>(ctx)
            .vortex_expect("list builder should canonicalize to listview");
        Canonical::List(listview)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use Nullability::NonNullable;
    use Nullability::Nullable;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::list::ListArraySlotsExt;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::ListViewBuilder;
    use crate::builders::builder_with_capacity;
    use crate::builders::list::ListArray;
    use crate::builders::list::ListBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::OffsetBuilderPType;
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

        let mut ctx = array_session().create_execution_ctx();
        let list_array = list.execute::<ListViewArray>(&mut ctx).unwrap();

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

        let mut ctx = array_session().create_execution_ctx();
        let list_array = list.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(list_array.list_elements_at(0).unwrap().len(), 3);
        assert_eq!(list_array.list_elements_at(1).unwrap().len(), 0);
        assert_eq!(list_array.list_elements_at(2).unwrap().len(), 3);
    }

    fn test_extend_builder_gen<O: OffsetBuilderPType>() {
        let list = ListArray::from_iter_opt_slow::<O, _, _>(
            [Some(vec![0, 1, 2]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )
        .unwrap()
        .into_array();
        assert_eq!(list.len(), 3);

        let mut ctx = array_session().create_execution_ctx();

        let mut builder = ListBuilder::<O>::with_capacity(Arc::new(I32.into()), Nullable, 18, 9);
        list.append_to_builder(&mut builder, &mut ctx).unwrap();
        list.append_to_builder(&mut builder, &mut ctx).unwrap();
        list.slice(0..0)
            .unwrap()
            .append_to_builder(&mut builder, &mut ctx)
            .unwrap();
        list.slice(1..3)
            .unwrap()
            .append_to_builder(&mut builder, &mut ctx)
            .unwrap();

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
        .into_array()
        .execute::<ListViewArray>(&mut ctx)
        .unwrap();

        let actual = builder.finish_into_canonical(&mut ctx).into_listview();

        assert_arrays_eq!(actual.elements(), expected.elements(), &mut ctx);

        assert_arrays_eq!(actual.offsets(), expected.offsets(), &mut ctx);

        assert!(
            actual
                .validity()
                .vortex_expect("list validity should be derivable")
                .mask_eq(
                    &expected
                        .validity()
                        .vortex_expect("list validity should be derivable"),
                    actual.len(),
                    &mut ctx,
                )
                .unwrap(),
        );
    }

    /// `append_to_builder` must handle any list builder kind without assuming the offset/size
    /// integer types produced by `builder_with_capacity`. It appends a `List`-encoded array and a
    /// `ListView`-encoded array into `ListViewBuilder`s and `ListBuilder`s with assorted (and
    /// non-`u64`) offset/size types.
    #[test]
    fn test_append_to_builder_any_list_builder() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let list = ListArray::from_iter_opt_slow::<u64, _, _>(
            [Some(vec![0, 1, 2]), None, Some(vec![4, 5])],
            Arc::new(I32.into()),
        )?
        .into_array();
        let listview = list
            .clone()
            .execute::<ListViewArray>(&mut ctx)?
            .into_array();
        let elem_dtype = || Arc::new(I32.into());

        // `builder_with_capacity` produces a `ListViewBuilder` for `DType::List`; appending the
        // `List`-encoded array must dispatch into it instead of bailing.
        let mut listview_builder = builder_with_capacity(list.dtype(), list.len());
        list.append_to_builder(listview_builder.as_mut(), &mut ctx)?;
        assert_arrays_eq!(listview_builder.finish(), list, &mut ctx);

        // A `ListViewBuilder` with non-`u64` (including signed) offset and size types must work
        // for both source encodings.
        let mut lv_u64_u32 =
            ListViewBuilder::<u64, u32>::with_capacity(elem_dtype(), Nullable, 8, 4);
        list.append_to_builder(&mut lv_u64_u32, &mut ctx)?;
        assert_arrays_eq!(lv_u64_u32.finish(), list, &mut ctx);

        let mut lv_i64_i32 =
            ListViewBuilder::<i64, i32>::with_capacity(elem_dtype(), Nullable, 8, 4);
        list.append_to_builder(&mut lv_i64_i32, &mut ctx)?;
        assert_arrays_eq!(lv_i64_i32.finish(), list, &mut ctx);

        let mut lv_u32_u32 =
            ListViewBuilder::<u32, u32>::with_capacity(elem_dtype(), Nullable, 8, 4);
        listview.append_to_builder(&mut lv_u32_u32, &mut ctx)?;
        assert_arrays_eq!(lv_u32_u32.finish(), list, &mut ctx);

        // Both source encodings appended into `ListBuilder`s with non-`u64` (including signed)
        // offset types.
        let mut list_builder = ListBuilder::<u32>::with_capacity(elem_dtype(), Nullable, 8, 4);
        list.append_to_builder(&mut list_builder, &mut ctx)?;
        assert_arrays_eq!(list_builder.finish(), list, &mut ctx);

        let mut list_builder_i32 = ListBuilder::<i32>::with_capacity(elem_dtype(), Nullable, 8, 4);
        listview.append_to_builder(&mut list_builder_i32, &mut ctx)?;
        assert_arrays_eq!(list_builder_i32.finish(), list, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_append_list_arrays_grow_builder() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Enough lists to exceed the offsets capacity of a zero-capacity builder, so appending
        // must grow the builder rather than panic in `uninit_range`.
        let lists: Vec<Option<Vec<i32>>> =
            (0..100).map(|i| (i % 10 != 0).then(|| vec![i])).collect();
        let source = ListArray::from_iter_opt_slow::<u32, _, _>(lists.clone(), Arc::clone(&dtype))?;
        let expected = ListArray::from_iter_opt_slow::<u32, _, _>(
            lists.iter().cloned().chain(lists.iter().cloned()),
            Arc::clone(&dtype),
        )?;

        // Appending twice checks growth from a non-empty builder and offset rebasing.
        let mut builder = ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), Nullable, 0, 0);
        builder.append_list_array(source.as_view(), &mut ctx)?;
        builder.append_list_array(source.as_view(), &mut ctx)?;
        assert_arrays_eq!(builder.finish(), expected, &mut ctx);

        let source_listview = source.into_array().execute::<ListViewArray>(&mut ctx)?;
        let mut builder = ListBuilder::<u32>::with_capacity(dtype, Nullable, 0, 0);
        builder.append_listview_array(source_listview.as_view(), &mut ctx)?;
        builder.append_listview_array(source_listview.as_view(), &mut ctx)?;
        assert_arrays_eq!(builder.finish(), expected, &mut ctx);

        Ok(())
    }

    /// A `ListArray`'s offsets can only describe contiguous, in-order lists, so an overlapping
    /// source has to be flattened before its elements can be appended in bulk. A sliced source,
    /// meanwhile, keeps the layout it has and is appended from wherever its first list starts.
    #[test]
    fn test_append_listview_array_flattens_overlaps_and_skips_leading_elements() -> VortexResult<()>
    {
        let mut ctx = array_session().create_execution_ctx();
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Overlapping source, so not zero-copyable to a list:
        // - List 0: [10, 20]
        // - List 1: null (size is intentionally non-zero in the source metadata)
        // - List 2: [10], sharing the elements list 0 already referenced
        let overlapping = unsafe {
            ListViewArray::new_unchecked(
                buffer![10i32, 20, 30].into_array(),
                buffer![0u32, 1, 0].into_array(),
                buffer![2u8, 2, 1].into_array(),
                Validity::from_iter([true, false, true]),
            )
        };
        assert!(!overlapping.is_zero_copy_to_list());

        // Zero-copyable source sliced past its first list, so its elements start at offset 2.
        let sliced = unsafe {
            ListViewArray::new_unchecked(
                buffer![40i32, 50, 60, 70].into_array(),
                buffer![0u32, 2].into_array(),
                buffer![2u32, 2].into_array(),
                Validity::AllValid,
            )
            .with_zero_copy_to_list(true)
        }
        .into_array()
        .slice(1..2)?
        .execute::<ListViewArray>(&mut ctx)?;

        let mut builder = ListBuilder::<u32>::with_capacity(dtype, Nullable, 0, 0);
        builder.append_listview_array(overlapping.as_view(), &mut ctx)?;
        builder.append_listview_array(sliced.as_view(), &mut ctx)?;

        let list = builder.finish_into_list();
        assert_arrays_eq!(
            list.elements(),
            PrimitiveArray::from_iter([10i32, 20, 10, 60, 70]),
            &mut ctx
        );
        assert_arrays_eq!(
            list.offsets(),
            PrimitiveArray::from_iter([0u32, 2, 2, 3, 5]),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_extend_builder() {
        test_extend_builder_gen::<i32>();
        test_extend_builder_gen::<i64>();

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

        let mut ctx = array_session().create_execution_ctx();
        let canon_values = chunked_list
            .unwrap()
            .as_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)
            .unwrap();

        assert_eq!(
            one_trailing_unused_element
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            canon_values
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            second_array
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            canon_values
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap()
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

        let mut ctx = array_session().create_execution_ctx();

        // Check actual values using scalar_at.

        let scalar0 = array.execute_scalar(0, &mut ctx).unwrap();
        let list0 = scalar0.as_list();
        assert_eq!(list0.len(), 2);
        if let Some(list0_items) = list0.elements() {
            assert_eq!(list0_items[0].as_primitive().typed_value::<i32>(), Some(1));
            assert_eq!(list0_items[1].as_primitive().typed_value::<i32>(), Some(2));
        }

        let scalar1 = array.execute_scalar(1, &mut ctx).unwrap();
        let list1 = scalar1.as_list();
        assert_eq!(list1.len(), 3);
        if let Some(list1_items) = list1.elements() {
            assert_eq!(list1_items[0].as_primitive().typed_value::<i32>(), Some(3));
            assert_eq!(list1_items[1].as_primitive().typed_value::<i32>(), Some(4));
            assert_eq!(list1_items[2].as_primitive().typed_value::<i32>(), Some(5));
        }

        let scalar2 = array.execute_scalar(2, &mut ctx).unwrap();
        let list2 = scalar2.as_list();
        assert!(list2.is_null()); // This should be null.

        // Check validity.
        assert!(
            array
                .validity()
                .vortex_expect("list validity should be derivable")
                .execute_is_valid(0, &mut ctx)
                .unwrap()
        );
        assert!(
            array
                .validity()
                .vortex_expect("list validity should be derivable")
                .execute_is_valid(1, &mut ctx)
                .unwrap()
        );
        assert!(
            !array
                .validity()
                .vortex_expect("list validity should be derivable")
                .execute_is_valid(2, &mut ctx)
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
        let mut ctx = array_session().create_execution_ctx();
        let mut builder =
            ListBuilder::<u32>::with_capacity(Arc::clone(&dtype), NonNullable, 20, 10);

        // Append a primitive array as a single list entry.
        let arr1 = buffer![1i32, 2, 3].into_array();
        builder.append_array_as_list(&arr1, &mut ctx).unwrap();

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
        builder.append_array_as_list(&arr2, &mut ctx).unwrap();

        // Append an empty array as a single list entry (empty list).
        let arr3 = buffer![0i32; 0].into_array();
        builder.append_array_as_list(&arr3, &mut ctx).unwrap();

        // Interleave with another list scalar (empty list).
        builder
            .append_value(Scalar::list_empty(Arc::clone(&dtype), NonNullable).as_list())
            .unwrap();

        let list = builder.finish_into_list();
        assert_eq!(list.len(), 5);

        // Verify elements array: [1, 2, 3, 10, 11, 4, 5].
        assert_arrays_eq!(
            list.elements(),
            PrimitiveArray::from_iter([1i32, 2, 3, 10, 11, 4, 5]),
            &mut ctx
        );

        // Verify offsets array.
        assert_arrays_eq!(
            list.offsets(),
            PrimitiveArray::from_iter([0u32, 3, 5, 7, 7, 7]),
            &mut ctx
        );

        // Test dtype mismatch error.
        let mut builder = ListBuilder::<u32>::with_capacity(dtype, NonNullable, 20, 10);
        let wrong_dtype_arr = buffer![1i64, 2, 3].into_array();
        assert!(
            builder
                .append_array_as_list(&wrong_dtype_arr, &mut ctx)
                .is_err()
        );
    }
}
