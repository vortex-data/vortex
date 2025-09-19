// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::{AsPrimitive, Zero};
use vortex_dtype::{
    DType, NativePType, Nullability, match_each_integer_ptype, match_each_native_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_err};

use crate::arrays::{ListArray, PrimitiveVTable};
use crate::builders::PrimitiveBuilder;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, Canonical, IntoArray, ToCanonical};

mod vtable;
pub use vtable::{ListViewEncoding, ListViewVTable};

#[cfg(test)]
mod tests;

mod compute;

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
/// use vortex_array::arrays::{ListViewArray, PrimitiveArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_buffer::buffer;
/// use std::sync::Arc;
///
/// // Create a list view array representing [[3, 4], [1], [2, 5]]
/// // Note: Unlike ListArray, offsets don't need to be monotonic
/// let elements = buffer![1i32, 2, 3, 4, 5].into_array();
/// let offsets = buffer![2u32, 0, 1].into_array();  // Out-of-order offsets
/// let sizes = buffer![2u32, 1, 2].into_array();  // Corresponding sizes
///
/// let list_view = ListViewArray::try_new(
///     elements.into_array(),
///     offsets.into_array(),
///     sizes.into_array(),
///     Validity::NonNullable,
/// ).unwrap();
///
/// assert_eq!(list_view.len(), 3);
///
/// // Access individual lists
/// let first_list = list_view.list_elements_at(0);
/// assert_eq!(first_list.len(), 2);
/// // First list contains elements[2..4] = [3, 4]
///
/// let first_offset = list_view.offset_at(0);
/// let first_size = list_view.size_at(0);
/// assert_eq!(first_offset, 2);
/// assert_eq!(first_size, 2);
/// ```
///
/// [`ListArray`]: crate::arrays::ListArray
#[derive(Clone, Debug)]
pub struct ListViewArray {
    /// The [`DType`] of the list array.
    ///
    /// This type **must** be the variant [`DType::List`].
    dtype: DType,

    /// The `elements` data array, where each list scalar is a _slice_ of the `elements` array, and
    /// each inner list element is a _scalar_ of the `elements` array.
    elements: ArrayRef,

    /// The `offsets` array indicating the start position of each list in elements.
    ///
    /// Since we also store `sizes`, this `offsets` field is allowed to be stored out-of-order
    /// (which is different from [`ListArray`](crate::arrays::ListArray)),
    offsets: ArrayRef,

    /// The `sizes` array indicating the length of each list.
    ///
    /// This field is intended to be paired with a corresponding offset to determine the list scalar
    /// we want to access.
    sizes: ArrayRef,

    /// The validity / null map of the array.
    ///
    /// Note that this null map refers to which list scalars are null, **not** which sub-elements of
    /// list scalars are null. The `elements` array will track individual value nullability.
    validity: Validity,

    /// The stats for this array.
    stats_set: ArrayStats,
}

impl ListViewArray {
    /// Get the length of the array.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.offsets.len(), self.sizes.len());
        self.offsets.len()
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new [`ListViewArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`ListViewArray::new_unchecked`].
    pub fn new(elements: ArrayRef, offsets: ArrayRef, sizes: ArrayRef, validity: Validity) -> Self {
        Self::try_new(elements, offsets, sizes, validity)
            .vortex_expect("ListViewArray construction failed")
    }

    /// Constructs a new `ListViewArray`.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants.
    pub fn try_new(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&elements, &offsets, &sizes, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(elements, offsets, sizes, validity) })
    }

    /// Creates a new [`ListViewArray`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - `offsets` and `sizes` must be non-nullable integer arrays.
    /// - `offsets` and `sizes` must have the same length.
    /// - Size integer width must be smaller than or equal to offset type (to prevent overflow).
    /// - For each `i`, `offsets[i] + sizes[i]` must not overflow and must be `<= elements.len()`.
    /// - If validity is an array, its length must equal `offsets.len()`.
    pub unsafe fn new_unchecked(
        elements: ArrayRef,
        offsets: ArrayRef,
        sizes: ArrayRef,
        validity: Validity,
    ) -> Self {
        Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), validity.nullability()),
            elements,
            offsets,
            sizes,
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`ListViewArray`].
    pub(crate) fn validate(
        elements: &dyn Array,
        offsets: &dyn Array,
        sizes: &dyn Array,
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
        let offset_ptype = offsets.dtype().as_ptype();
        let size_ptype = sizes.dtype().as_ptype();
        vortex_ensure!(
            size_ptype.byte_width() <= offset_ptype.byte_width(),
            "size type {:?} must fit within offset type {:?}",
            size_ptype,
            offset_ptype
        );

        // Validate the `offsets` and `sizes` arrays.
        match_each_integer_ptype!(offset_ptype, |O| {
            match_each_integer_ptype!(size_ptype, |S| {
                let offsets_primitive = offsets.to_primitive();
                let sizes_primitive = sizes.to_primitive();

                let offsets_slice = offsets_primitive.as_slice::<O>();
                let sizes_slice = sizes_primitive.as_slice::<S>();

                validate_offsets_and_sizes::<O, S>(
                    offsets_slice,
                    sizes_slice,
                    elements.len() as u64,
                )?;
            })
        });

        // If a validity array is present, it must be the same length as the ListView.
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == offsets.len(),
                "validity with size {validity_len} does not match array size {}",
                offsets.len()
            );
        }

        Ok(())
    }

    /// Returns the offset at the given index.
    pub fn offset_at(&self, index: usize) -> usize {
        assert!(
            index < self.len(),
            "Index {index} out of bounds 0..{}",
            self.len()
        );

        // Fast path for `PrimitiveArray`.
        self.offsets
            .as_opt::<PrimitiveVTable>()
            .map(|p| match_each_native_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                // Slow path: use `scalar_at` if we can't downcast directly to `PrimitiveArray`.
                self.offsets
                    .scalar_at(index)
                    .as_primitive()
                    .as_::<usize>()
                    .vortex_expect("offset must fit in usize")
            })
    }

    /// Returns the size at the given index.
    pub fn size_at(&self, index: usize) -> usize {
        assert!(
            index < self.len(),
            "Index {} out of bounds 0..{}",
            index,
            self.len()
        );

        // Fast path for `PrimitiveArray`.
        self.sizes
            .as_opt::<PrimitiveVTable>()
            .map(|p| match_each_native_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                // Slow path: use `scalar_at` if we can't downcast directly to `PrimitiveArray`.
                self.sizes
                    .scalar_at(index)
                    .as_primitive()
                    .as_::<usize>()
                    .vortex_expect("size must fit in usize")
            })
    }

    /// Returns the elements at the given index from the list array.
    pub fn list_elements_at(&self, index: usize) -> ArrayRef {
        let offset = self.offset_at(index);
        let size = self.size_at(index);
        self.elements().slice(offset..offset + size)
    }

    /// Returns the offsets array.
    pub fn offsets(&self) -> &ArrayRef {
        &self.offsets
    }

    /// Returns the sizes array.
    pub fn sizes(&self) -> &ArrayRef {
        &self.sizes
    }

    /// Returns the elements array.
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }
}

/// Helper function to validate `offsets` and `sizes` with specific types.
fn validate_offsets_and_sizes<O, S>(
    offsets_slice: &[O],
    sizes_slice: &[S],
    elements_len: u64,
) -> VortexResult<()>
where
    O: NativePType + PartialOrd + Zero,
    S: NativePType + PartialOrd + Zero,
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

        vortex_ensure!(
            end <= elements_len,
            "offset[{i}] + size[{i}] = {end} exceeds elements length {elements_len}",
        );
    }

    Ok(())
}

/// Create a [`ListViewArray`] from a [`ListArray`](crate::arrays::ListArray) by computing `sizes`
/// from `offsets`.
pub fn list_view_from_list(list: ListArray) -> ListViewArray {
    // TODO(connor)[ListView]: Create a version of `Canonical::empty` for `ListView`. It might
    // also be worth specializing that for all canonical encodings.
    // If the list is empty, create an empty `ListView` with the same offset dtype as the input.
    if list.is_empty() {
        let empty_offsets = Canonical::empty(list.offsets().dtype()).into_array();
        let empty_sizes = Canonical::empty(list.offsets().dtype()).into_array();
        let empty_validity = list.validity().clone();

        // SAFETY: Everything is empty so all the variants are satisfied.
        return unsafe {
            ListViewArray::new_unchecked(
                list.elements().clone(),
                empty_offsets,
                empty_sizes,
                empty_validity,
            )
        };
    }

    let len = list.len();

    // Get the `offsets` array directly from the `ListArray` (preserving its type).
    let list_offsets = list.offsets().clone();

    // We need to slice the `offsets` to remove the last element (`ListArray` has n+1 offsets).
    let adjusted_offsets = list_offsets.slice(0..len);

    // Create sizes array by computing differences between consecutive offsets.
    // Use the same dtype as the offsets array to ensure compatibility.
    let sizes = match_each_integer_ptype!(list_offsets.dtype().as_ptype(), |P| {
        let mut sizes_builder = PrimitiveBuilder::<P>::with_capacity(Nullability::NonNullable, len);

        // Create uninit range for direct memory access.
        let mut sizes_range = sizes_builder.uninit_range(len);

        // Compute sizes as the difference between consecutive offsets.
        for i in 0..len {
            let start = list.offset_at(i);
            let end = list.offset_at(i + 1);
            let size = end - start;

            // Set size value directly without creating scalar.
            sizes_range.set_value(
                i,
                P::try_from(size).vortex_expect("size must fit in offset type"),
            );
        }

        // SAFETY: We have initialized all values in the range.
        unsafe {
            sizes_range.finish();
        }

        sizes_builder.finish_into_primitive().into_array()
    });

    // SAFETY: Since everything came from an existing valid `ListArray`, and the `sizes` were
    // derived from valid and in-order `offsets`, we know these fields are valid.
    unsafe {
        ListViewArray::new_unchecked(
            list.elements().clone(),
            adjusted_offsets,
            sizes,
            list.validity().clone(),
        )
    }
}
