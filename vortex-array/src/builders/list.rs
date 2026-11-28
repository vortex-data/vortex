// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
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
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(value_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::new_in(value_dtype, nullability, BufferAllocatorRef::static_ref())
    }

    /// Creates a new `ListBuilder` with the default capacity using `allocator`.
    pub fn new_in(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_in(
            value_dtype,
            nullability,
            DEFAULT_BUILDER_CAPACITY,
            allocator,
        )
    }

    /// Create a new [`ListArray`] builder with the given `capacity`.
    ///
    /// # Notes
    ///
    /// The number of offsets is one more than the length (# of list scalars) in the array.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        _elements_capacity: usize,
        capacity: usize,
    ) -> Self {
        Self::with_capacity_in(
            value_dtype,
            nullability,
            capacity,
            BufferAllocatorRef::static_ref(),
        )
    }

    /// Creates a list builder with the given capacity using `allocator`.
    pub fn with_capacity_in(
        value_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        let elements_builder =
            ChildBuilder::with_capacity(value_dtype.as_ref(), 2 * capacity, allocator);
        let mut offsets_builder =
            PrimitiveBuilder::<O>::with_capacity_in(NonNullable, capacity + 1, allocator);

        // The first offset is always 0 and represents an empty list.
        offsets_builder.append_zero();

        Self {
            elements_builder,
            offsets_builder,
            nulls: ValidityBuilder::new(capacity, allocator),
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
        vortex_ensure_eq!(array.dtype(), self.element_dtype());

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

    /// Finishes the builder directly into a [`ListArray`].
    pub fn finish_into_list(&mut self) -> ListArray {
        assert_eq!(
            self.offsets_builder.len(),
            self.nulls.len() + 1,
            "offsets length must be one more than nulls length."
        );

        let elements = self.elements_builder.finish();
        let offsets = self.offsets_builder.finish();
        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        // Re-seed the retained offsets builder with the leading 0 for the next batch.
        self.offsets_builder.append_zero();

        // SAFETY:
        // - The offsets come from a non-nullable integer builder seeded with a leading zero, so
        //   they are a non-nullable integer array of at least one non-negative value.
        // - Every append records `elements_builder.len()` after appending its elements, so the
        //   offsets increase monotonically and the last one is exactly the elements length.
        // - The assert above pairs one validity entry with each list.
        unsafe { ListArray::new_unchecked(elements, offsets, validity) }
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
        self.offsets_builder.append_n_values(
            O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            n,
        );
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        let curr_len = self.elements_builder.len();
        // A list with a null element can be a list with a zero-span offset and a cleared validity
        // bit.
        self.offsets_builder.append_n_values(
            O::from_usize(curr_len).vortex_expect("Failed to convert from usize to <O>"),
            n,
        );
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "ListBuilder received a scalar with the wrong dtype"
        );

        self.append_value(scalar.as_list())
    }

    fn reserve_chunks(&mut self, additional: usize) {
        self.elements_builder.reserve_chunks(additional);
        self.nulls.reserve_runs(additional);
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
    use std::sync::LazyLock;

    use Nullability::NonNullable;
    use Nullability::Nullable;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::list::ListArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::list::ListArray;
    use crate::builders::list::ListBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::executor::VortexSessionExecute;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// A `ListArray`'s offsets can only describe contiguous, in-order lists, so an overlapping
    /// source has to be flattened before its elements can be appended in bulk. A sliced source,
    /// meanwhile, keeps the layout it has and is appended from wherever its first list starts.
    #[test]
    fn test_append_listview_array_flattens_overlaps_and_skips_leading_elements() -> VortexResult<()>
    {
        let mut ctx = SESSION.create_execution_ctx();
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

        let mut builder = ListBuilder::<u32>::with_capacity_in(
            dtype,
            Nullable,
            0,
            BufferAllocatorRef::static_ref(),
        );
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

        let expected = ListArray::new(
            buffer![10i32, 20, 10, 60, 70].into_array(),
            buffer![0u32, 2, 2, 3, 5].into_array(),
            Validity::from_iter([true, false, true, true]),
            &mut ctx,
        );
        assert_arrays_eq!(list, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_array_with_gap() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let one_trailing_unused_element = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?;
        let second_array = ListArray::try_new(
            buffer![5i32, 6].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?;
        let chunked_list = ChunkedArray::try_new(
            vec![
                one_trailing_unused_element.into_array(),
                second_array.into_array(),
            ],
            DType::List(Arc::new(DType::Primitive(I32, NonNullable)), NonNullable),
        )?;

        let expected = ListArray::new(
            buffer![1i32, 2, 3, 5, 6].into_array(),
            buffer![0u32, 3, 5].into_array(),
            Validity::NonNullable,
            &mut ctx,
        );
        assert_arrays_eq!(chunked_list, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_append_array_as_list_wrong_dtype() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = ListBuilder::<u32>::with_capacity_in(
            Arc::new(I32.into()),
            NonNullable,
            4,
            BufferAllocatorRef::static_ref(),
        );
        assert!(
            builder
                .append_array_as_list(&buffer![1i64, 2, 3].into_array(), &mut ctx)
                .is_err()
        );
    }
}
