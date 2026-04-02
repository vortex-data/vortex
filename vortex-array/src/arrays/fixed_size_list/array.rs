// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::FixedSizeList;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::validity::Validity;

/// The `elements` data array, where each fixed-size list scalar is a _slice_ of the `elements`
/// array, and each inner list element is a _scalar_ of the `elements` array.
///
/// The fixed-size list scalars are contiguous (regardless of nullability for easy lookups),
/// each with equal size in memory.
pub(super) const ELEMENTS_SLOT: usize = 0;
/// The validity / null map of the array.
///
/// Note that this null map refers to which fixed-size list scalars are null, **not** which
/// sub-elements of fixed-size list scalars are null. The `elements` array will track individual
/// value nullability.
pub(super) const VALIDITY_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["elements", "validity"];

/// The canonical encoding for fixed-size list arrays.
///
/// A fixed-size list array stores lists where each list has the same number of elements. This is
/// similar to a 2D array or matrix where the inner dimension is fixed.
///
/// ## Data Layout
///
/// Unlike [`ListArray`] which uses offsets, `FixedSizeListArray` stores elements contiguously and
/// uses a fixed `list_size`:
///
/// - **Elements array**: A flat array containing all list elements concatenated together
/// - **List size**: The fixed number of elements in each list
/// - **Validity**: Optional mask indicating which lists are null
///
/// The list at index `i` contains elements from `elements[i * list_size..(i + 1) * list_size]`.
///
/// [`ListArray`]: crate::arrays::ListArray
///
/// # Examples
///
/// ```
/// # fn main() -> vortex_error::VortexResult<()> {
/// use vortex_array::arrays::{FixedSizeListArray, PrimitiveArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_buffer::buffer;
///
/// // Create a fixed-size list array representing [[1, 2] [3, 4], [5, 6], [7, 8]]
/// let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
/// let list_size = 2;
///
/// let fixed_list_array = FixedSizeListArray::new(
///     elements.into_array(),
///     list_size,
///     Validity::NonNullable,
///     4, // 4 lists
/// );
///
/// assert_eq!(fixed_list_array.len(), 4);
/// assert_eq!(fixed_list_array.list_size(), 2);
///
/// // Access individual lists
/// let first_list = fixed_list_array.fixed_size_list_elements_at(0)?;
/// assert_eq!(first_list.len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct FixedSizeListData {
    /// The [`DType`] of the fixed-size list.
    ///
    /// This type **must** be the variant [`DType::FixedSizeList`].
    pub(super) dtype: DType,

    /// Slots holding [elements].
    pub(super) slots: Vec<Option<ArrayRef>>,

    /// The size of each fixed-size list scalar in the array.
    ///
    /// We store the size of each fixed-size list in the array as a field for convenience.
    list_size: u32,

    /// The length of the array.
    ///
    /// Note that this is different from the size of each fixed-size list scalar (`list_size`).
    ///
    /// The main reason we need to store this (rather than calculate it on the fly via `list_size`
    /// and `elements.len()`) is because in the degenerate case where `list_size == 0`, we cannot
    /// use `0 / 0` to determine the length.
    pub(super) len: usize,

    /// The stats for this array.
    pub(super) stats_set: ArrayStats,
}

impl FixedSizeListData {
    /// Creates a new `FixedSizeListArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `FixedSizeListArray::new_unchecked`.
    pub fn new(elements: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        Self::try_new(elements, list_size, validity, len)
            .vortex_expect("FixedSizeListArray construction failed")
    }

    /// Constructs a new `FixedSizeListArray`.
    ///
    /// See `FixedSizeListArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented
    /// in `FixedSizeListArray::new_unchecked`.
    pub fn try_new(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<Self> {
        Self::validate(&elements, len, list_size, &validity)?;

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe { Self::new_unchecked(elements, list_size, validity, len) })
    }

    /// Creates a new `FixedSizeListArray` without validation from these components:
    ///
    /// * `elements` is the data array where each fixed-size list is a slice.
    /// * `list_size` is the fixed number of elements in each list.
    /// * `validity` holds the null values.
    /// * `len` is the number of lists in the array.
    ///
    /// # Safety
    ///
    /// The inputs are **valid** if:
    ///
    /// - The `Validity` length (if it exists) times the `list_size` is equal to the length of the
    ///   `elements` (or put another way, the length of the array divided by the size of each
    ///   fixed-size list is equal to the length of the validity).
    /// - The length of the `elements` array is equal to the length of the outer array times the
    ///   `list_size` (`elements.len() == list_size * len`).
    pub unsafe fn new_unchecked(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&elements, len, list_size, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `FixedSizeListArray` parameters");

        let nullability = validity.nullability();
        let validity_slot = validity_to_child(&validity, len);

        Self {
            dtype: DType::FixedSizeList(Arc::new(elements.dtype().clone()), list_size, nullability),
            slots: vec![Some(elements), validity_slot],
            list_size,
            len,
            stats_set: Default::default(),
        }
    }

    pub fn into_parts(mut self) -> (ArrayRef, Validity, DType) {
        let validity = self.validity();
        (
            self.slots[ELEMENTS_SLOT]
                .take()
                .vortex_expect("FixedSizeListArray elements slot"),
            validity,
            self.dtype,
        )
    }

    /// Validates the components that would be used to create a `FixedSizeListArray`.
    ///
    /// This function checks all the invariants required by `FixedSizeListArray::new_unchecked`.
    pub fn validate(
        elements: &ArrayRef,
        len: usize,
        list_size: u32,
        validity: &Validity,
    ) -> VortexResult<()> {
        // If a validity array is present, it must be the same length as the fixed-size list array.
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                len == validity_len,
                InvalidArgument: "validity with size {validity_len} does not match fixed-size list array size {len}",
            );
        }

        // A fixed-size list array where each list scalar is empty is completely useless, but we can
        // support it regardless.
        if list_size == 0 {
            vortex_ensure!(
                elements.is_empty(),
                InvalidArgument: "a degenerate (`list_size == 0`) `FixedSizeList` should have no underlying elements"
            );
            return Ok(());
        }

        vortex_ensure!(
            len * list_size as usize == elements.len(),
            InvalidArgument: "the `elements` array has the incorrect number of elements to construct a \
                `FixedSizeList[{list_size}] array of length {len}",
        );

        Ok(())
    }

    /// Returns the dtype of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the validity of the array.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> Validity {
        child_to_validity(&self.slots[VALIDITY_SLOT], self.dtype.nullability())
    }

    /// Returns the validity as a [`Mask`](vortex_mask::Mask).
    pub fn validity_mask(&self) -> vortex_mask::Mask {
        self.validity().to_mask(self.len())
    }

    /// Returns the elements array.
    pub fn elements(&self) -> &ArrayRef {
        self.slots[ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("FixedSizeListArray elements slot")
    }

    /// The size of each fixed-size list scalar in the array.
    pub const fn list_size(&self) -> u32 {
        self.list_size
    }
}

impl Array<FixedSizeList> {
    /// Creates a new `FixedSizeListArray`.
    pub fn new(elements: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        Array::try_from_data(FixedSizeListData::new(elements, list_size, validity, len))
            .vortex_expect("FixedSizeListData is always valid")
    }

    /// Constructs a new `FixedSizeListArray`.
    pub fn try_new(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<Self> {
        Array::try_from_data(FixedSizeListData::try_new(
            elements, list_size, validity, len,
        )?)
    }

    /// Creates a new `FixedSizeListArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`FixedSizeListData::new_unchecked`].
    pub unsafe fn new_unchecked(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> Self {
        Array::try_from_data(unsafe {
            FixedSizeListData::new_unchecked(elements, list_size, validity, len)
        })
        .vortex_expect("FixedSizeListData is always valid")
    }
}

impl FixedSizeListData {
    pub fn fixed_size_list_elements_at(&self, index: usize) -> VortexResult<ArrayRef> {
        debug_assert!(
            index < self.len,
            "index {} out of bounds: the len is {}",
            index,
            self.len,
        );
        debug_assert!(self.validity().is_valid(index).unwrap_or(false));

        let start = self.list_size as usize * index;
        let end = self.list_size as usize * (index + 1);
        self.elements().slice(start..end)
    }
}
