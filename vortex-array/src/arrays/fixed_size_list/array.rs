// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::FixedSizeList;
use crate::dtype::DType;
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
/// use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
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
    /// The length of the array.
    ///
    /// Note that this is different from the size of each fixed-size list scalar (`list_size`).
    ///
    /// The main reason we need to store this (rather than calculate it on the fly via `list_size`
    /// and `elements.len()`) is because in the degenerate case where `list_size == 0`, we cannot
    /// use `0 / 0` to determine the length.
    pub(super) degenerate_len: usize,
}

impl Display for FixedSizeListData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "degenerate_len: {}", self.degenerate_len)
    }
}

pub struct FixedSizeListDataParts {
    pub elements: ArrayRef,
    pub validity: Validity,
    pub dtype: DType,
}

impl FixedSizeListData {
    pub(crate) fn make_slots(elements: &ArrayRef, validity: &Validity, len: usize) -> ArraySlots {
        smallvec![Some(elements.clone()), validity_to_child(validity, len)]
    }

    /// Creates a new `FixedSizeListArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `FixedSizeListArray::new_unchecked`.
    pub fn build(elements: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        Self::try_build(elements, list_size, validity, len)
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
    pub(crate) fn try_build(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<Self> {
        Self::validate(&elements, len, list_size, &validity)?;

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe { Self::new_unchecked(list_size, len) })
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
    pub unsafe fn new_unchecked(list_size: u32, len: usize) -> Self {
        Self {
            degenerate_len: if list_size == 0 { len } else { 0 },
        }
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
}

pub trait FixedSizeListArrayExt: TypedArrayRef<FixedSizeList> {
    fn dtype_parts(&self) -> (&DType, u32, crate::dtype::Nullability) {
        match self.as_ref().dtype() {
            DType::FixedSizeList(element_dtype, list_size, nullability) => {
                (element_dtype.as_ref(), *list_size, *nullability)
            }
            _ => unreachable!("FixedSizeListArrayExt requires a fixed-size list dtype"),
        }
    }

    fn elements(&self) -> &ArrayRef {
        self.as_ref().slots()[ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("FixedSizeListArray elements slot")
    }

    fn list_size(&self) -> u32 {
        let (_, list_size, _) = self.dtype_parts();
        list_size
    }

    fn fixed_size_list_validity(&self) -> Validity {
        let (_, _, nullability) = self.dtype_parts();
        child_to_validity(self.as_ref().slots()[VALIDITY_SLOT].as_ref(), nullability)
    }

    fn fixed_size_list_elements_at(&self, index: usize) -> VortexResult<ArrayRef> {
        debug_assert!(
            index < self.as_ref().len(),
            "index {} out of bounds: the len is {}",
            index,
            self.as_ref().len(),
        );
        #[expect(clippy::debug_assert_with_mut_call)]
        {
            debug_assert!(
                self.fixed_size_list_validity()
                    .execute_is_valid(index, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap_or(false)
            );
        }

        let start = self.list_size() as usize * index;
        let end = self.list_size() as usize * (index + 1);
        self.elements().slice(start..end)
    }
}
impl<T: TypedArrayRef<FixedSizeList>> FixedSizeListArrayExt for T {}

impl Array<FixedSizeList> {
    /// Creates a new `FixedSizeListArray`.
    pub fn new(elements: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        let dtype = DType::FixedSizeList(
            Arc::new(elements.dtype().clone()),
            list_size,
            validity.nullability(),
        );
        let slots = FixedSizeListData::make_slots(&elements, &validity, len);
        let data = FixedSizeListData::build(elements, list_size, validity, len);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FixedSizeList, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Constructs a new `FixedSizeListArray`.
    pub fn try_new(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<Self> {
        let dtype = DType::FixedSizeList(
            Arc::new(elements.dtype().clone()),
            list_size,
            validity.nullability(),
        );
        let slots = FixedSizeListData::make_slots(&elements, &validity, len);
        let data = FixedSizeListData::try_build(elements, list_size, validity, len)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FixedSizeList, dtype, len, data).with_slots(slots),
            )
        })
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
        let dtype = DType::FixedSizeList(
            Arc::new(elements.dtype().clone()),
            list_size,
            validity.nullability(),
        );
        let slots = FixedSizeListData::make_slots(&elements, &validity, len);
        let data = unsafe { FixedSizeListData::new_unchecked(list_size, len) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FixedSizeList, dtype, len, data).with_slots(slots),
            )
        }
    }

    pub fn into_data_parts(self) -> FixedSizeListDataParts {
        let dtype = self.dtype().clone();
        let elements = self.slots()[ELEMENTS_SLOT]
            .clone()
            .vortex_expect("FixedSizeListArray elements slot");
        let validity = self.fixed_size_list_validity();
        FixedSizeListDataParts {
            elements,
            validity,
            dtype,
        }
    }
}
