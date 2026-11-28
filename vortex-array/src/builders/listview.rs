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

use num_traits::ToPrimitive;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::IntoArray;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::builders::ArrayBuilder;
use crate::builders::ChildBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::PrimitiveBuilder;
use crate::builders::UninitRange;
use crate::builders::ValidityBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::OffsetBuilderPType;
use crate::match_each_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;

/// A builder for creating [`ListViewArray`] instances, parameterized by the [`OffsetBuilderPType`]
/// of the `offsets` and the `sizes` builders.
///
/// This builder tracks both offsets and sizes using potentially different integer types for memory
/// efficiency. For example, you might use `u64` for offsets but only `u8` for sizes if your lists
/// are small.
///
/// Any combination of [`OffsetBuilderPType`] types is valid, as long as the type of `sizes` can fit
/// into the type of `offsets`.
pub struct ListViewBuilder<O: OffsetBuilderPType, S: OffsetBuilderPType> {
    /// The [`DType`] of the [`ListViewArray`]. This **must** be a [`DType::List`].
    dtype: DType,

    /// The builder for the underlying elements of the [`ListArray`](crate::arrays::ListArray).
    elements_builder: ChildBuilder,

    /// The builder for the `offsets` into the `elements` array.
    offsets_builder: PrimitiveBuilder<O>,

    /// The builder for the `sizes` of each list view.
    sizes_builder: PrimitiveBuilder<S>,

    /// The null map builder of the [`ListViewArray`].
    nulls: ValidityBuilder,

    /// Whether the appends so far leave the result zero-copyable to a [`ListArray`].
    ///
    /// Only [`append_listview_array`](Self::append_listview_array) can clear this; every other
    /// append writes its lists back to back.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    zero_copy_to_list: bool,
}

impl<O: OffsetBuilderPType, S: OffsetBuilderPType> ListViewBuilder<O, S> {
    /// Creates a new `ListViewBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    #[deprecated(note = "use `new_in` with an explicit allocator")]
    pub fn new(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::new_in(element_dtype, nullability, BufferAllocatorRef::static_ref())
    }

    /// Creates a new `ListViewBuilder` with the default capacity using `allocator`.
    pub fn new_in(
        element_dtype: Arc<DType>,
        nullability: Nullability,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        Self::with_capacity_in(
            element_dtype,
            nullability,
            DEFAULT_BUILDER_CAPACITY,
            allocator,
        )
    }

    /// Create a new [`ListViewArray`] builder with the given `capacity`.
    ///
    /// # Panics
    ///
    /// Panics if the size type `S` cannot fit within the offset type `O`.
    #[deprecated(note = "use `with_capacity_in` with an explicit allocator")]
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        nullability: Nullability,
        _elements_capacity: usize,
        capacity: usize,
    ) -> Self {
        Self::with_capacity_in(
            element_dtype,
            nullability,
            capacity,
            BufferAllocatorRef::static_ref(),
        )
    }

    /// Creates a list-view builder with the given capacity using `allocator`.
    ///
    /// # Panics
    ///
    /// Panics if the size type `S` cannot fit within the offset type `O`.
    pub fn with_capacity_in(
        element_dtype: Arc<DType>,
        nullability: Nullability,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> Self {
        let elements_builder = ChildBuilder::with_capacity(&element_dtype, 2 * capacity, allocator);

        let offsets_builder =
            PrimitiveBuilder::<O>::with_capacity_in(Nullability::NonNullable, capacity, allocator);
        let sizes_builder =
            PrimitiveBuilder::<S>::with_capacity_in(Nullability::NonNullable, capacity, allocator);

        let nulls = ValidityBuilder::new(capacity, allocator);

        Self {
            dtype: DType::List(element_dtype, nullability),
            elements_builder,
            offsets_builder,
            sizes_builder,
            nulls,
            zero_copy_to_list: true,
        }
    }

    /// Appends an array as a single non-null list entry to the builder.
    ///
    /// The input `array` must have the same dtype as the element dtype of this list builder.
    ///
    /// Note that the list entry will be non-null but the elements themselves are allowed to be null
    /// (only if the elements [`DType`] is nullable, of course).
    pub fn append_array_as_list(
        &mut self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(array.dtype(), self.element_dtype());

        let curr_offset = self.elements_builder.len();
        let num_elements = array.len();

        // We must assert this even in release mode to ensure that the safety comment in
        // `finish_into_listview` is correct.
        assert!(
            ((curr_offset + num_elements) as u64) < O::max_value_as_u64(),
            "appending this list would cause an offset overflow"
        );

        self.elements_builder.append_array(array, ctx)?;
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

    /// Appends `array` as `n` identical non-null lists, storing its elements once.
    ///
    /// A `ListViewArray` can point many views at one range of elements, so a repeated list costs
    /// its elements once however many rows it covers. The elements go in as one appended array, so
    /// a caller that hands over the same `array` on every call - a sparse array filling the gaps
    /// between its patches, say - stores those elements once for the whole result.
    ///
    /// The views share their elements, so the result is no longer zero-copyable to a
    /// [`ListArray`](crate::arrays::ListArray) unless it covers a single row or empty lists.
    pub fn append_array_as_repeated_list(
        &mut self,
        array: &ArrayRef,
        n: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(array.dtype(), self.element_dtype());

        if n == 0 {
            return Ok(());
        }

        let curr_offset = self.elements_builder.len();
        let num_elements = array.len();

        // We must assert this even in release mode to ensure that the safety comment in
        // `finish_into_listview` is correct.
        assert!(
            ((curr_offset + num_elements) as u64) < O::max_value_as_u64(),
            "appending this list would cause an offset overflow"
        );

        self.elements_builder.append_array(array, ctx)?;

        let offset =
            O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`");
        let size = S::from_usize(num_elements).vortex_expect("Failed to convert from usize to `S`");
        self.offsets_builder.append_n_values(offset, n);
        self.sizes_builder.append_n_values(size, n);
        self.nulls.append_n_non_nulls(n);

        if n > 1 && num_elements > 0 {
            self.zero_copy_to_list = false;
        }

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

        let zero_copy_to_list = std::mem::replace(&mut self.zero_copy_to_list, true);

        // SAFETY:
        // - Both the offsets and the sizes are non-nullable.
        // - The offsets, sizes, and validity have the same length since we always appended the same
        //   amount.
        // - We checked on construction that the sizes type fits into the offsets.
        // - In every method that adds values to this builder (`append_value`, `append_scalar`,
        //   `append_list_array`, and `append_listview_array`), we checked that `offset + size`
        //   does not overflow. `append_listview_array` rebases the offsets it was handed onto
        //   exactly the elements it appended, so the source's bound carries over.
        // - Every append writes its lists back to back, so the result is zero-copyable to a
        //   `ListArray` unless `zero_copy_to_list` recorded an appended layout we left alone.
        unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, validity)
                .with_zero_copy_to_list(zero_copy_to_list)
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

    /// Appends the values of a [`List`]-encoded `array` to this builder.
    ///
    /// List encodings dispatch here through
    /// [`match_each_list_builder!`](crate::match_each_list_builder) because the concrete list
    /// builders are generic over their offset/size integer types, which cannot be named through a
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

        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |OffsetType| {
            extend_from_list(
                self,
                array.elements(),
                offsets.as_slice::<OffsetType>(),
                ctx,
            )?
        });
        Ok(())
    }

    /// Appends the values of a [`ListView`]-encoded `array` to this builder.
    ///
    /// See [`append_list_array`](Self::append_list_array); this is the same hook for the canonical
    /// [`ListViewArray`] encoding.
    ///
    /// The views keep the layout they arrived in, so overlapping sources keep sharing their
    /// elements and the finished array reports [`is_zero_copy_to_list`] as `false`. Callers that
    /// need an exact layout should [`rebuild`](ListViewArray::rebuild) it.
    ///
    /// [`is_zero_copy_to_list`]: crate::arrays::listview::ListViewData::is_zero_copy_to_list
    pub fn append_listview_array(
        &mut self,
        array: ArrayView<'_, ListView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }

        let len = array.len();

        // Materialize the metadata once and do the trimming and the rebase by hand. Going through
        // `rebuild(ListViewRebuildMode::TrimElements)` would subtract the window start from every
        // offset with a compute kernel, only for the rebase below to add this builder's elements
        // base straight back on - two passes, one of them through the compute stack, for one
        // addition per offset. Casting the sizes to the builder's type is another kernel for what
        // is a copy.
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let sizes = array.sizes().clone().execute::<PrimitiveArray>(ctx)?;

        // The window of `elements` that the views actually reference. Everything outside it is
        // unreachable and must not be appended. An exact source covers its window back to back and
        // in order, so the first and last view bound it; any other layout has to be searched.
        let (start, end) = if array.is_zero_copy_to_list() {
            let last = len - 1;
            (
                metadata_at(&offsets, 0),
                metadata_at(&offsets, last) + metadata_at(&sizes, last),
            )
        } else {
            array.into_owned().referenced_element_bounds(ctx)?
        };

        // An exact source references every element it carries, back to back, so it lands flush
        // against the elements already in the builder. Any other layout does not.
        self.zero_copy_to_list &= array.is_zero_copy_to_list();

        self.nulls.append_validity(array.validity()?, len);

        // Bulk append the referenced elements; the offsets are rebased onto them below.
        let elements_base = self.elements_builder.len();

        // We must assert this even in release mode to ensure that the safety comment in
        // `finish_into_listview` is correct.
        assert!(
            ((elements_base + (end - start)) as u64) < O::max_value_as_u64(),
            "appending this list would cause an offset overflow"
        );

        if end > start {
            self.elements_builder
                .append_array(&array.elements().slice(start..end)?, ctx)?;
        }

        // Every view lies inside `start..end`, so rebasing it onto `elements_base` keeps it inside
        // the elements this builder now holds - which is what lets `finish_into_listview` build the
        // array unchecked.
        assert_eq!(
            self.elements_builder.len(),
            elements_base + (end - start),
            "appending the referenced elements did not extend the child by the window's length"
        );

        self.offsets_builder.reserve_exact(len);
        let offsets_range = self.offsets_builder.uninit_range(len);
        match_each_integer_ptype!(offsets.ptype(), |A| {
            extend_rebased_offsets::<O, A>(
                offsets_range,
                offsets.as_slice::<A>(),
                start,
                elements_base,
            );
        });

        self.sizes_builder.reserve_exact(len);
        let mut sizes_range = self.sizes_builder.uninit_range(len);
        if sizes.ptype() == S::PTYPE {
            // The sizes already have the builder's type, so there is nothing to convert.
            sizes_range.copy_from_slice(0, sizes.as_slice::<S>());
            // SAFETY: `copy_from_slice` initialized all `len` values, and the sizes builder is
            // non-nullable.
            unsafe { sizes_range.finish() };
        } else {
            match_each_integer_ptype!(sizes.ptype(), |A| {
                extend_converted_sizes::<S, A>(sizes_range, sizes.as_slice::<A>());
            });
        }

        Ok(())
    }
}

impl<O: OffsetBuilderPType, S: OffsetBuilderPType> ArrayBuilder for ListViewBuilder<O, S> {
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
        self.offsets_builder.append_n_values(
            O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
            n,
        );
        self.sizes_builder.append_n_values(S::zero(), n);

        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        debug_assert_eq!(self.offsets_builder.len(), self.sizes_builder.len());
        debug_assert_eq!(self.offsets_builder.len(), self.nulls.len());

        // Get the current position in the elements array.
        let curr_offset = self.elements_builder.len();

        // A null list can have any representation, but we choose to use the zero representation.
        self.offsets_builder.append_n_values(
            O::from_usize(curr_offset).vortex_expect("Failed to convert from usize to `O`"),
            n,
        );
        self.sizes_builder.append_n_values(S::zero(), n);

        // This is the only difference from `append_zeros`.
        self.nulls.append_n_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            scalar.dtype(),
            self.dtype(),
            "ListViewBuilder received a scalar with the wrong dtype"
        );

        let list_scalar = scalar.as_list();
        self.append_value(list_scalar)
    }

    fn reserve_chunks(&mut self, additional: usize) {
        self.elements_builder.reserve_chunks(additional);
        self.nulls.reserve_runs(additional);
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.elements_builder.reserve_exact(capacity * 2);
        self.offsets_builder.reserve_exact(capacity);
        self.sizes_builder.reserve_exact(capacity);
        self.nulls.reserve_exact(capacity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_listview().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::List(self.finish_into_listview())
    }
}

/// Appends `ListArray`-layout lists (`n + 1` cumulative offsets) into a [`ListViewBuilder`].
///
/// Lists in a `ListArray` are contiguous, so the referenced elements can be appended in bulk,
/// with the offsets rebased onto the builder's elements and the sizes taken from consecutive
/// offset differences.
fn extend_from_list<O, S, OffsetType>(
    builder: &mut ListViewBuilder<O, S>,
    elements: &ArrayRef,
    offsets: &[OffsetType],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    O: OffsetBuilderPType,
    S: OffsetBuilderPType,
    OffsetType: IntegerPType,
{
    let num_lists = offsets.len() - 1;
    let first: usize = offsets[0].as_();
    let last: usize = offsets[num_lists].as_();

    let elements_base = builder.elements_builder.len();

    // We must assert this even in release mode to ensure that the safety comment in
    // `finish_into_listview` is correct.
    assert!(
        ((elements_base + (last - first)) as u64) < O::max_value_as_u64(),
        "appending this list would cause an offset overflow"
    );

    if last > first {
        builder
            .elements_builder
            .append_array(&elements.slice(first..last)?, ctx)?;
    }

    builder.offsets_builder.reserve_exact(num_lists);
    builder.sizes_builder.reserve_exact(num_lists);
    let mut offsets_range = builder.offsets_builder.uninit_range(num_lists);
    let mut sizes_range = builder.sizes_builder.uninit_range(num_lists);
    for i in 0..num_lists {
        let start: usize = offsets[i].as_();
        let end: usize = offsets[i + 1].as_();
        offsets_range.set_value(
            i,
            O::from_usize(start - first + elements_base)
                .vortex_expect("Failed to convert from usize to `O`"),
        );
        sizes_range.set_value(
            i,
            S::from_usize(end - start).vortex_expect("Failed to convert from usize to `S`"),
        );
    }
    // SAFETY: We have initialized all `num_lists` values in both ranges, and both the `offsets`
    // and the `sizes` builders are non-nullable.
    unsafe { offsets_range.finish() };
    unsafe { sizes_range.finish() };
    Ok(())
}

/// Reads one non-nullable integer value of list view metadata as a `usize`.
fn metadata_at(metadata: &PrimitiveArray, index: usize) -> usize {
    match_each_integer_ptype!(metadata.ptype(), |A| {
        metadata.as_slice::<A>()[index]
            .to_usize()
            .vortex_expect("list view metadata must fit in a usize")
    })
}

/// Writes `offsets` into `range`, moved off the source's element window and onto the elements the
/// builder already holds.
///
/// `window_start` is the first element the source's views reference, so every offset is at least
/// `window_start` and the subtraction cannot underflow.
fn extend_rebased_offsets<O: OffsetBuilderPType, A: IntegerPType>(
    mut range: UninitRange<O>,
    offsets: &[A],
    window_start: usize,
    elements_base: usize,
) {
    debug_assert_eq!(range.len(), offsets.len());

    for (i, &offset) in offsets.iter().enumerate() {
        let offset = offset
            .to_usize()
            .vortex_expect("offsets must always fit in usize");
        debug_assert!(
            offset >= window_start,
            "offset {offset} precedes the referenced window at {window_start}"
        );
        let rebased = O::from_usize(offset - window_start + elements_base)
            .vortex_expect("rebased offset did not fit into the builder's offset type");
        range.set_value(i, rebased);
    }

    // SAFETY: We have set all the values in the range, and since `offsets` are non-nullable, we are
    // done.
    unsafe { range.finish() };
}

/// Writes `sizes` into `range`, converting them to the builder's size type.
///
/// Sizes that already have the builder's type are copied in bulk by the caller instead.
fn extend_converted_sizes<S: OffsetBuilderPType, A: IntegerPType>(
    mut range: UninitRange<S>,
    sizes: &[A],
) {
    debug_assert_eq!(range.len(), sizes.len());

    for (i, &size) in sizes.iter().enumerate() {
        let size = S::from_usize(
            size.to_usize()
                .vortex_expect("sizes must always fit in usize"),
        )
        .vortex_expect("size did not fit into the builder's size type");
        range.set_value(i, size);
    }

    // SAFETY: We have set all the values in the range, and since `sizes` are non-nullable, we are
    // done.
    unsafe { range.finish() };
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ListViewBuilder;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::listview::ListViewArrayExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Empty lists are appended in bulk rather than one offset/size pair at a time, so a large
    /// `n` must land at the current end of `elements` (preserving `zero_copy_to_list` monotonic
    /// offsets), with validity taken from which method was called.
    #[rstest]
    #[case::zeros(false)]
    #[case::nulls(true)]
    fn test_bulk_empty_lists(#[case] null: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity_in(
            dtype,
            Nullable,
            0,
            BufferAllocatorRef::static_ref(),
        );

        // Put something in `elements` first, so the empty lists have a non-zero offset to share.
        builder.append_array_as_list(&buffer![1i32, 2].into_array(), &mut ctx)?;

        const N: usize = 5;
        if null {
            builder.append_nulls(N);
        } else {
            builder.append_zeros(N);
        }

        let listview = builder.finish_into_listview();
        assert_eq!(listview.len(), N + 1);
        assert!(listview.is_zero_copy_to_list());
        for i in 1..=N {
            assert_eq!(listview.size_at(i, &mut ctx), 0);
            assert_eq!(listview.offset_at(i, &mut ctx), 2);
            assert_eq!(listview.validity()?.execute_is_valid(i, &mut ctx)?, !null);
        }
        Ok(())
    }

    /// A constant list array points every view at a single copy of the value; flattening it in the
    /// builder would materialize a copy per row.
    #[test]
    fn test_constant_list_append_keeps_one_copy_of_the_value() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let element_dtype: Arc<DType> = Arc::new(I32.into());

        const ROWS: usize = 10_000;
        let fill = Scalar::list(
            Arc::clone(&element_dtype),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            NonNullable,
        );
        let constant = ConstantArray::new(fill, ROWS).into_array();

        let mut builder = ListViewBuilder::<u64, u64>::with_capacity_in(
            element_dtype,
            NonNullable,
            0,
            BufferAllocatorRef::static_ref(),
        );
        constant.append_to_builder(&mut builder, &mut ctx)?;
        let listview = builder.finish_into_listview();

        assert_eq!(listview.len(), ROWS);
        assert_eq!(
            listview.elements().len(),
            3,
            "the fill value should be stored once, not once per row",
        );
        assert!(!listview.is_zero_copy_to_list());
        assert_arrays_eq!(&listview.into_array(), &constant, &mut ctx);
        Ok(())
    }

    /// Only the elements the views reference land in the builder: appending a slice out of the
    /// middle of a list array leaves the elements on either side of it behind.
    #[test]
    fn test_append_listview_array_trims_unreferenced_elements() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Five lists of two elements each, of which we append only the middle three.
        let source = ListArray::from_iter_slow::<u32, _>(
            (0..5).map(|i| vec![2 * i, 2 * i + 1]),
            Arc::clone(&dtype),
        )?
        .into_array()
        .execute::<ListViewArray>(&mut ctx)?;
        let middle = source.slice(1..4)?.execute::<ListViewArray>(&mut ctx)?;

        let mut builder = ListViewBuilder::<u32, u32>::with_capacity_in(
            Arc::clone(&dtype),
            NonNullable,
            0,
            BufferAllocatorRef::static_ref(),
        );
        builder.append_listview_array(middle.as_view(), &mut ctx)?;
        // A second append has to rebase onto the elements already in the builder.
        builder.append_listview_array(middle.as_view(), &mut ctx)?;
        let listview = builder.finish_into_listview();

        assert_eq!(
            listview.elements().len(),
            12,
            "only the six referenced elements of each append should have landed",
        );
        assert!(
            listview.is_zero_copy_to_list(),
            "trimming an exact source keeps the result exact",
        );

        let expected = ListArray::from_iter_slow::<u32, _>(
            (1..4).chain(1..4).map(|i| vec![2 * i, 2 * i + 1]),
            dtype,
        )?;
        assert_arrays_eq!(listview, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_extend_from_array_overlapping_listview() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype: Arc<DType> = Arc::new(I32.into());

        // Non-ZCTL source:
        // - List 0: [10, 20]
        // - List 1: null (size is intentionally non-zero in source metadata)
        // - List 2: [10]
        let source = unsafe {
            ListViewArray::new_unchecked(
                buffer![10i32, 20, 30].into_array(),
                buffer![0u32, 1, 0].into_array(),
                buffer![2u8, 2, 1].into_array(),
                Validity::from_iter([true, false, true]),
            )
        };
        assert!(!source.is_zero_copy_to_list());

        let mut builder = ListViewBuilder::<u32, u32>::with_capacity_in(
            dtype,
            Nullable,
            0,
            BufferAllocatorRef::static_ref(),
        );
        builder.append_listview_array(source.as_view(), &mut ctx)?;

        let listview = builder.finish_into_listview();
        // The builder kept the source's overlapping layout.
        assert!(!listview.is_zero_copy_to_list());
        assert_eq!(listview.offset_at(0, &mut ctx), 0);
        assert_eq!(listview.offset_at(2, &mut ctx), 0);
        // List 1 is null, so the builder keeps its source size without rewriting it to zero.
        assert_eq!(listview.size_at(1, &mut ctx), source.size_at(1, &mut ctx));

        let expected = ListArray::new(
            buffer![10i32, 20, 10].into_array(),
            buffer![0u32, 2, 2, 3].into_array(),
            Validity::from_iter([true, false, true]),
            &mut ctx,
        );
        assert_arrays_eq!(listview, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_error_append_null_to_non_nullable() {
        let dtype: Arc<DType> = Arc::new(I32.into());
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity_in(
            Arc::clone(&dtype),
            NonNullable,
            0,
            BufferAllocatorRef::static_ref(),
        );

        let null_scalar = Scalar::null(DType::List(dtype, Nullable));
        let err = builder
            .append_value(null_scalar.as_list())
            .expect_err("null value into non-nullable builder must fail");
        assert!(err.to_string().contains("null value to non-nullable"));
    }

    #[test]
    fn test_append_array_as_list_wrong_dtype() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = ListViewBuilder::<u32, u32>::with_capacity_in(
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
