// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DynArray;
use crate::ToCanonical;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::validity_to_child;

pub(super) const ELEMENTS_SLOT: usize = 0;
pub(super) const OFFSETS_SLOT: usize = 1;
pub(super) const SIZES_SLOT: usize = 2;
pub(super) const VALIDITY_SLOT: usize = 3;
pub(super) const NUM_SLOTS: usize = 4;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["elements", "offsets", "sizes", "validity"];

/// The canonical encoding for variable-length list arrays.
///
/// The `ListViewArray` encoding differs from [`ListArray`] in that it stores a child `sizes` array
/// in addition to a child `offsets` array (which is the _only_ child in [`ListArray`]).
///
/// In the past, we used [`ListArray`] as the canonical encoding for [`DType::List`], but we have
/// since migrated to `ListViewArray` for a few reasons:
///
/// - Enables better SIMD vectorization (no sequential dependency when reading `offsets`)
/// - Allows out-of-order offsets for better compression (we can shuffle the buffers)
/// - Supports different integer types for offsets vs sizes
///
/// It is worth mentioning that this encoding mirrors Apache Arrow's `ListView` array type, but does
/// not exactly mirror the similar type found in DuckDB and Velox, which stores the pair of offset
/// and size in a row-major fashion rather than column-major. More specifically, the row-major
/// layout has a single child array with alternating offset and size next to each other.
///
/// We choose the column-major layout as it allows better compressability, as well as using
/// different (logical) integer widths for our `offsets` and `sizes` buffers (note that the
/// compressor will likely compress to a different bit-packed width, but this is speaking strictly
/// about flexibility in the logcial type).
///
/// # Examples
///
/// ```
/// # fn main() -> vortex_error::VortexResult<()> {
/// # use vortex_array::arrays::{ListViewArray, PrimitiveArray};
/// # use vortex_array::validity::Validity;
/// # use vortex_array::IntoArray;
/// # use vortex_buffer::buffer;
/// # use std::sync::Arc;
/// #
/// // Create a list view array representing [[3, 4], [1], [2, 3]].
/// // Note: Unlike `ListArray`, offsets don't need to be monotonic.
///
/// let elements = buffer![1i32, 2, 3, 4, 5].into_array();
/// let offsets = buffer![2u32, 0, 1].into_array();  // Out-of-order offsets
/// let sizes = buffer![2u32, 1, 2].into_array();  // The sizes cause overlaps
///
/// let list_view = ListViewArray::new(
///     elements.into_array(),
///     offsets.into_array(),
///     sizes.into_array(),
///     Validity::NonNullable,
/// );
///
/// assert_eq!(list_view.len(), 3);
///
/// // Access individual lists
/// let first_list = list_view.list_elements_at(0)?;
/// assert_eq!(first_list.len(), 2);
/// // First list contains elements[2..4] = [3, 4]
///
/// let first_offset = list_view.offset_at(0);
/// let first_size = list_view.size_at(0);
/// assert_eq!(first_offset, 2);
/// assert_eq!(first_size, 2);
/// # Ok(())
/// # }
/// ```
///
/// [`ListArray`]: crate::arrays::ListArray
#[derive(Clone, Debug)]
pub struct ListViewArray {
    /// The [`DType`] of the list array.
    ///
    /// This type **must** be the variant [`DType::List`].
    pub(super) dtype: DType,

    /// Slots holding [elements, offsets, sizes].
    pub(super) slots: Vec<Option<ArrayRef>>,

    // TODO(connor)[ListView]: Add the n+1 memory allocation optimization.
    /// A flag denoting if the array is zero-copyable* to a [`ListArray`](crate::arrays::ListArray).
    ///
    /// We use this information to help us more efficiently rebuild / compact our data.
    ///
    /// When this flag is true (indicating sorted offsets with no gaps and no overlaps and all
    /// `offsets[i] + sizes[i]` are in order), conversions can bypass the very expensive rebuild
    /// process which must rebuild the array from scratch.
    is_zero_copy_to_list: bool,

    /// The validity / null map of the array.
    ///
    /// Note that this null map refers to which list scalars are null, **not** which sub-elements of
    /// list scalars are null. The `elements` array will track individual value nullability.
    pub(super) validity: Validity,

    /// The stats for this array.
    pub(super) stats_set: ArrayStats,
}

pub struct ListViewArrayParts {
    pub elements_dtype: Arc<DType>,

    /// See `ListViewArray::elements`
    pub elements: ArrayRef,

    /// See `ListViewArray::offsets`
    pub offsets: ArrayRef,

    /// See `ListViewArray::sizes`
    pub sizes: ArrayRef,

    /// See `ListViewArray::validity`
    pub validity: Validity,
}

impl ListViewArray {
    /// Creates a new [`ListViewArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`ListViewArray::new_unchecked`].
    pub fn new(elements: ArrayRef, offsets: ArrayRef, sizes: ArrayRef, validity: Validity) -> Self {
        Self::try_new(elements, offsets, sizes, validity)
            .vortex_expect("`ListViewArray` construction failed")
    }

    /// Constructs a new `ListViewArray`.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented
    /// in [`ListViewArray::new_unchecked`].
    pub fn try_new(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&elements, &offsets, &sizes, &validity)?;

        let len = offsets.len();
        let validity_slot = validity_to_child(&validity, len);

        Ok(Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), validity.nullability()),
            slots: vec![Some(elements), Some(offsets), Some(sizes), validity_slot],
            validity,
            is_zero_copy_to_list: false,
            stats_set: Default::default(),
        })
    }

    /// Creates a new [`ListViewArray`] without validation.
    ///
    /// This unsafe function does not check the validity of the data. Prefer calling [`new()`] or
    /// [`try_new()`] over this function, as they will check the validity of the data.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    /// [`new()`]: Self::new
    /// [`try_new()`]: Self::try_new
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - `offsets` and `sizes` must be non-nullable integer arrays.
    /// - `offsets` and `sizes` must have the same length.
    /// - Size integer width must be smaller than or equal to offset type (to prevent overflow).
    /// - For each `i`, `offsets[i] + sizes[i]` must not overflow and must be `<= elements.len()`
    ///   (even if the corresponding view is defined as null by the validity array).
    /// - If validity is an array, its length must equal `offsets.len()`.
    pub unsafe fn new_unchecked(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::validate(&elements, &offsets, &sizes, &validity)
                .vortex_expect("Failed to crate `ListViewArray`");
        }

        let len = offsets.len();
        let validity_slot = validity_to_child(&validity, len);

        Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), validity.nullability()),
            slots: vec![Some(elements), Some(offsets), Some(sizes), validity_slot],
            validity,
            is_zero_copy_to_list: false,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`ListViewArray`].
    pub fn validate(
        elements: &ArrayRef,
        offsets: &ArrayRef,
        sizes: &ArrayRef,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Check that offsets and sizes are integer arrays and non-nullable.
        vortex_ensure!(
            offsets.dtype().is_int() && !offsets.dtype().is_nullable(),
            "offsets must be non-nullable integer array, got {}",
            offsets.dtype()
        );
        vortex_ensure!(
            sizes.dtype().is_int() && !sizes.dtype().is_nullable(),
            "sizes must be non-nullable integer array, got {}",
            sizes.dtype()
        );

        // Check that they have the same length.
        vortex_ensure!(
            offsets.len() == sizes.len(),
            "offsets and sizes must have the same length, got {} and {}",
            offsets.len(),
            sizes.len()
        );

        // Check that the size type can fit within the offset type to prevent overflows.
        let size_ptype = sizes.dtype().as_ptype();
        let offset_ptype = offsets.dtype().as_ptype();

        // If a validity array is present, it must be the same length as the `ListViewArray`.
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == offsets.len(),
                "validity with size {validity_len} does not match array size {}",
                offsets.len()
            );
        }

        // Skip host-only validation when offsets/sizes are not host-resident.
        if offsets.is_host() && sizes.is_host() {
            let offsets_primitive = offsets.to_primitive();
            let sizes_primitive = sizes.to_primitive();

            // Validate the `offsets` and `sizes` arrays.
            match_each_integer_ptype!(offset_ptype, |O| {
                match_each_integer_ptype!(size_ptype, |S| {
                    let offsets_slice = offsets_primitive.as_slice::<O>();
                    let sizes_slice = sizes_primitive.as_slice::<S>();

                    validate_offsets_and_sizes::<O, S>(
                        offsets_slice,
                        sizes_slice,
                        elements.len() as u64,
                    )?;
                })
            });
        }

        Ok(())
    }

    /// Sets whether this [`ListViewArray`] is zero-copyable to a [`ListArray`].
    ///
    /// This is an optimization flag that enables more efficient conversion to [`ListArray`] without
    /// needing to copy or reorganize the data.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    ///
    /// # Safety
    ///
    /// When setting `is_zctl` to `true`, the caller must ensure that the [`ListViewArray`] is
    /// actually zero-copyable to a [`ListArray`]. This means:
    ///
    /// - Offsets must be sorted (but not strictly sorted, zero-length lists are allowed).
    /// - `offsets[i] + sizes[i] == offsets[i + 1]` for all `i`.
    /// - No gaps in elements between first and last referenced elements.
    /// - No overlapping list views (each element referenced at most once).
    ///
    /// Note that leading and trailing unreferenced elements **ARE** allowed.
    pub unsafe fn with_zero_copy_to_list(mut self, is_zctl: bool) -> Self {
        if cfg!(debug_assertions) && is_zctl {
            validate_zctl(
                self.elements(),
                self.offsets().to_primitive(),
                self.sizes().to_primitive(),
            )
            .vortex_expect("Failed to validate zero-copy to list flag");
        }
        self.is_zero_copy_to_list = is_zctl;
        self
    }

    /// Verifies that the `ListViewArray` is zero-copyable to a [`ListArray`].
    ///
    /// This will run an expensive validation of the `ListViewArray`'s components. It will check the
    /// following things:
    ///
    /// - Offsets must be sorted (but not strictly sorted, zero-length lists are allowed).
    /// - No gaps in elements between first and last referenced elements.
    /// - No overlapping list views (each element referenced at most once).
    ///
    /// Note that leading and trailing unreferenced elements **ARE** allowed.
    ///
    /// This method should really only be called if the caller knows that the `ListViewArray` will
    /// be converted into a [`ListArray`] in the future, and the caller wants to set the
    /// optimization flag to `true` with the unsafe [`with_zero_copy_to_list`] method.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    /// [`with_zero_copy_to_list`]: Self::with_zero_copy_to_list
    pub fn verify_is_zero_copy_to_list(&self) -> bool {
        validate_zctl(
            self.elements(),
            self.offsets().to_primitive(),
            self.sizes().to_primitive(),
        )
        .is_ok()
    }

    pub fn into_parts(mut self) -> ListViewArrayParts {
        let dtype = self.dtype.into_list_element_opt().vortex_expect("is list");
        ListViewArrayParts {
            elements_dtype: dtype,
            elements: self.slots[ELEMENTS_SLOT]
                .take()
                .vortex_expect("ListViewArray elements slot"),
            offsets: self.slots[OFFSETS_SLOT]
                .take()
                .vortex_expect("ListViewArray offsets slot"),
            sizes: self.slots[SIZES_SLOT]
                .take()
                .vortex_expect("ListViewArray sizes slot"),
            validity: self.validity,
        }
    }

    /// Returns the offset at the given index.
    ///
    /// Note that it is possible the corresponding list view is null (which is only defined by the
    /// validity map). Regardless, we are still guaranteed that this offset is valid by the
    /// invariants of [`ListViewArray`].
    pub fn offset_at(&self, index: usize) -> usize {
        assert!(
            index < self.len(),
            "Index {index} out of bounds 0..{}",
            self.len()
        );

        // Fast path for `PrimitiveArray`.
        self.offsets()
            .as_opt::<Primitive>()
            .map(|p| match_each_integer_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                // Slow path: use `scalar_at` if we can't downcast directly to `PrimitiveArray`.
                self.offsets()
                    .scalar_at(index)
                    .vortex_expect("offsets must support scalar_at")
                    .as_primitive()
                    .as_::<usize>()
                    .vortex_expect("offset must fit in usize")
            })
    }

    /// Returns the size at the given index.
    ///
    /// Note that it is possible the corresponding list view is null (which is only defined by the
    /// validity map). Regardless, we are still guaranteed that this size is valid by the invariants
    /// of [`ListViewArray`].
    pub fn size_at(&self, index: usize) -> usize {
        assert!(
            index < self.len(),
            "Index {} out of bounds 0..{}",
            index,
            self.len()
        );

        // Fast path for `PrimitiveArray`.
        self.sizes()
            .as_opt::<Primitive>()
            .map(|p| match_each_integer_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                // Slow path: use `scalar_at` if we can't downcast directly to `PrimitiveArray`.
                self.sizes()
                    .scalar_at(index)
                    .vortex_expect("sizes must support scalar_at")
                    .as_primitive()
                    .as_::<usize>()
                    .vortex_expect("size must fit in usize")
            })
    }

    /// Returns the elements at the given index from the list array.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice operation fails.
    pub fn list_elements_at(&self, index: usize) -> VortexResult<ArrayRef> {
        let offset = self.offset_at(index);
        let size = self.size_at(index);
        self.elements().slice(offset..offset + size)
    }

    /// Returns the offsets array.
    pub fn offsets(&self) -> &ArrayRef {
        self.slots[OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray offsets slot")
    }

    /// Returns the sizes array.
    pub fn sizes(&self) -> &ArrayRef {
        self.slots[SIZES_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray sizes slot")
    }

    /// Returns the elements array.
    pub fn elements(&self) -> &ArrayRef {
        self.slots[ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("ListViewArray elements slot")
    }

    /// Returns true if the `ListViewArray` is zero-copyable to a
    /// [`ListArray`](crate::arrays::ListArray).
    pub fn is_zero_copy_to_list(&self) -> bool {
        self.is_zero_copy_to_list
    }
}

/// Helper function to validate `offsets` and `sizes` with specific types.
fn validate_offsets_and_sizes<O, S>(
    offsets_slice: &[O],
    sizes_slice: &[S],
    elements_len: u64,
) -> VortexResult<()>
where
    O: IntegerPType,
    S: IntegerPType,
{
    debug_assert_eq!(offsets_slice.len(), sizes_slice.len());

    #[allow(clippy::absurd_extreme_comparisons, unused_comparisons)]
    for i in 0..offsets_slice.len() {
        let offset = offsets_slice[i];
        let size = sizes_slice[i];

        vortex_ensure!(offset >= O::zero(), "cannot have negative offsets");
        vortex_ensure!(size >= S::zero(), "cannot have negative size");

        let offset_u64 = offset
            .to_u64()
            .ok_or_else(|| vortex_err!("offset[{i}] = {offset:?} cannot be converted to u64"))?;

        let size_u64 = size
            .to_u64()
            .ok_or_else(|| vortex_err!("size[{i}] = {size:?} cannot be converted to u64"))?;

        // Check for overflow when adding offset + size.
        let end = offset_u64.checked_add(size_u64).ok_or_else(|| {
            vortex_err!("offset[{i}] ({offset_u64}) + size[{i}] ({size_u64}) would overflow u64")
        })?;

        if offset_u64 == elements_len {
            vortex_ensure!(
                size_u64 == 0,
                "views to the end of the elements array (length {elements_len}) must have size 0 \
                    (had size {size_u64})"
            );
        }

        vortex_ensure!(
            end <= elements_len,
            "offset[{i}] + size[{i}] = {offset_u64} + {size_u64} = {end} \
            exceeds elements length {elements_len}",
        );
    }

    Ok(())
}

/// Helper function to validate if the [`ListViewArray`] components are actually zero-copyable to
/// [`ListArray`](crate::arrays::ListArray).
fn validate_zctl(
    elements: &ArrayRef,
    offsets_primitive: PrimitiveArray,
    sizes_primitive: PrimitiveArray,
) -> VortexResult<()> {
    // Offsets must be sorted (but not strictly sorted, zero-length lists are allowed), even
    // if there are null views.
    if let Some(is_sorted) = offsets_primitive.statistics().compute_is_sorted() {
        vortex_ensure!(is_sorted, "offsets must be sorted");
    } else {
        vortex_bail!("offsets must report is_sorted statistic");
    }

    // Validate that offset[i] + size[i] <= offset[i+1] for all items
    // This ensures views are non-overlapping and properly ordered for zero-copy-to-list
    fn validate_monotonic_ends<O: IntegerPType, S: IntegerPType>(
        offsets_slice: &[O],
        sizes_slice: &[S],
        len: usize,
    ) -> VortexResult<()> {
        let mut max_end = 0usize;

        for i in 0..len {
            let offset = offsets_slice[i].to_usize().unwrap_or(usize::MAX);
            let size = sizes_slice[i].to_usize().unwrap_or(usize::MAX);

            // Check that this view starts at or after the previous view ended
            vortex_ensure!(
                offset >= max_end,
                "Zero-copy-to-list requires views to be non-overlapping and ordered: \
                 view[{}] starts at {} but previous views extend to {}",
                i,
                offset,
                max_end
            );

            // Update max_end for the next iteration
            let end = offset.saturating_add(size);
            max_end = max_end.max(end);
        }

        Ok(())
    }

    let offsets_dtype = offsets_primitive.dtype();
    let sizes_dtype = sizes_primitive.dtype();
    let len = offsets_primitive.len();

    // Check that offset + size values are monotonic (no overlaps)
    match_each_integer_ptype!(offsets_dtype.as_ptype(), |O| {
        match_each_integer_ptype!(sizes_dtype.as_ptype(), |S| {
            let offsets_slice = offsets_primitive.as_slice::<O>();
            let sizes_slice = sizes_primitive.as_slice::<S>();

            validate_monotonic_ends(offsets_slice, sizes_slice, len)?;
        })
    });

    // TODO(connor)[ListView]: Making this allocation is expensive, but the more efficient
    // implementation would be even more complicated than this. We could use a bit buffer denoting
    // if positions in `elements` are used, and then additionally store a separate flag that tells
    // us if a position is used more than once.
    let mut element_references = vec![0u8; elements.len()];

    fn count_references<O: IntegerPType, S: IntegerPType>(
        element_references: &mut [u8],
        offsets_primitive: PrimitiveArray,
        sizes_primitive: PrimitiveArray,
    ) {
        let offsets_slice = offsets_primitive.as_slice::<O>();
        let sizes_slice = sizes_primitive.as_slice::<S>();

        // Note that we ignore nulls here, as the "null" view metadata must still maintain the same
        // invariants as non-null views, even for a `bool` information.
        for i in 0..offsets_slice.len() {
            let offset: usize = offsets_slice[i].as_();
            let size: usize = sizes_slice[i].as_();
            for j in offset..offset + size {
                element_references[j] = element_references[j].saturating_add(1);
            }
        }
    }

    match_each_integer_ptype!(offsets_primitive.ptype(), |O| {
        match_each_integer_ptype!(sizes_primitive.ptype(), |S| {
            count_references::<O, S>(&mut element_references, offsets_primitive, sizes_primitive);
        })
    });

    // Allow leading and trailing unreferenced elements, but not gaps in the middle.
    let leftmost_used = element_references
        .iter()
        .position(|&references| references != 0);
    let rightmost_used = element_references
        .iter()
        .rposition(|&references| references != 0);

    if let (Some(first_ref), Some(last_ref)) = (leftmost_used, rightmost_used) {
        vortex_ensure!(
            element_references[first_ref..=last_ref]
                .iter()
                .all(|&references| references != 0),
            "found gap in elements array between first and last referenced elements"
        );
    }

    vortex_ensure!(element_references.iter().all(|&references| references <= 1));

    Ok(())
}
