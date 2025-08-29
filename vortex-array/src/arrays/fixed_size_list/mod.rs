// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};

use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::{Array, ArrayRef};

mod vtable;
pub use vtable::{FixedSizeListEncoding, FixedSizeListVTable};

#[cfg(test)]
mod tests;

/// The canonical encoding for fixed-size list arrays.
#[derive(Clone, Debug)]
pub struct FixedSizeListArray {
    /// The [`DType`] of the fixed-size list.
    ///
    /// This type **must** be the variant [`DType::FixedSizeList`].
    dtype: DType,

    /// The `elements` data array, where each fixed-size list scalar is a _slice_ of the `elements`
    /// array, and each inner list element is a _scalar_ of the `elements` array.
    ///
    /// The fixed-size list scalars (or the elements of the array) are contiguous (regardless of
    /// nullability for easy lookups), each with equal size in memory.
    elements: ArrayRef,

    /// The size of each fixed-size list scalar in the array.
    ///
    /// We store the size of each fixed-size list in the array as a field for convenience.
    list_size: u32,

    /// The validity / null map of the array.
    ///
    /// Note that this null map refers to the fixed-size list scalars, **not** the elements of the
    /// _individual_ fixed-size list scalars. The `elements` array will track individual value
    /// nullability.
    validity: Validity,

    /// The length of the array.
    ///
    /// Note that this is different from the size of each fixed-size list scalar (`list_size`).
    ///
    /// The main reason we need to store this (rather than calculate it on the fly via `list_size`
    /// and `elements.len()`) is because in the degenerate case where `list_size == 0`, we cannot
    /// use `0 / 0` to determine the length.
    len: usize,

    /// The stats for this array.
    stats_set: ArrayStats,
}

impl FixedSizeListArray {
    /// Creates a new `FixedSizeListArray`. This is simply a wrapper around [`try_new()`].
    ///
    /// # Panics
    ///
    /// Panics if the inputs are invalid. See
    ///
    /// [`try_new()`]: Self::try_new
    pub fn new(elements: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        Self::try_new(elements, list_size, validity, len)
            .vortex_expect("FixedSizeListArray `try_new` failed")
    }

    /// Attempts to create a new `FixedSizeListArray`.
    ///
    /// # Errors
    ///
    /// Returns an error if the inputs are invalid. The inputs are **valid** if:
    ///
    /// - The `list_size` is 0 and:
    ///   - The `elements` array is empty.
    ///   - The `len` is equal to the length of the `validity` map.
    /// - The length of the `elements` array is a multiple of the size of the fixed-size lists
    ///   (`list_size`).
    /// - The `Validity` length (if it exists) times the `list_size` is equal to the length of the
    ///   `elements` (or put another way, the length of the array divided by the size of each
    ///   fixed-size list is equal to the length of the validity).
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

    /// Creates a new `FixedSizeListArray`, assuming that the caller has validated the inputs.
    ///
    /// # Safety
    ///
    /// This function is only safe to call if the inputs are valid. See [`try_new()`] for more
    /// details on what the validity requirements are.
    ///
    /// [`try_new()`]: Self::try_new
    pub unsafe fn new_unchecked(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> Self {
        let nullability = validity.nullability();

        Self {
            dtype: DType::FixedSizeList(Arc::new(elements.dtype().clone()), list_size, nullability),
            elements,
            list_size,
            validity,
            len,
            stats_set: Default::default(),
        }
    }

    /// Returns the elements array.
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// The size of each fixed-size list scalar in the array.
    pub const fn list_size(&self) -> u32 {
        self.list_size
    }

    /// Returns the elements at the given index from the list array.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn fixed_size_list_at(&self, index: usize) -> ArrayRef {
        debug_assert!(
            index < self.len,
            "index out of bounds: the len is {} but the index is {index}",
            self.len
        );
        debug_assert!(self.validity.is_valid(index));

        let start = self.list_size as usize * index;
        let end = self.list_size as usize * (index + 1);
        self.elements().slice(start..end)
    }

    /// Checks if the components of a `FixedSizeListArray` are valid.
    ///
    /// See [`try_new()`](Self::try_new) for the validation semantics.
    fn validate(
        elements: &dyn Array,
        len: usize,
        list_size: u32,
        validity: &Validity,
    ) -> VortexResult<()> {
        // A fixed-size list array where each list scalar is empty is completely useless, but we can
        // support it regardless.
        if list_size == 0 {
            vortex_ensure!(
                elements.is_empty() && validity.maybe_len().is_none_or(|vlen| vlen == len),
                "an empty `FixedSizeList` should have no elements"
            );
            return Ok(());
        }

        let num_elements = elements.len();

        vortex_ensure!(
            len * list_size as usize == num_elements,
            "the `elements` array has the incorrect number of elements to construct a \
                `FixedSizeList[{list_size}] array of length {len}",
        );

        // If a validity array is present, it must be the same length as the fixed-size list array.
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                len == validity_len,
                "validity with size {validity_len} does not match fixed-size list array size {len}",
            );
        }

        Ok(())
    }
}
