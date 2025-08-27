// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused)] // TODO(connor)[FixedSizeList]: Remove this!

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};

use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::{Array, ArrayRef};

mod vtable;
pub use vtable::{FixedSizeListEncoding, FixedSizeListVTable};

/// The canonical encoding for fixed-size list arrays.
#[derive(Clone, Debug)]
pub struct FixedSizeListArray {
    /// The [`DType`] of the fixed-size list.
    ///
    /// This type **must** be the variant [`DType::FixedSizeList`].
    dtype: DType,

    /// The values data array, where each fixed-size list scalar is a slice of this `values` array.
    ///
    /// The fixed-size list scalars (or the elements of the array) are contiguous (regardless of
    /// nullability for easy lookups), each with equal size in memory.
    values: ArrayRef,

    /// The size of each fixed-size list scalar in the array.
    ///
    /// We store the size of each fixed-size list in the array as a field for convenience.
    list_size: u32,

    /// The validity / null map of the array.
    ///
    /// Note that this null map refers to the fixed-size list scalars, **not** the elements of the
    /// _individual_ fixed-size list scalars. The `values` array will track individual value
    /// nullability.
    validity: Validity,

    // Would it be a good idea to make this a const generic parameter?
    /// The length of the array.
    ///
    /// Note that this is different from the size of each fixed-size list scalar (`list_size`).
    ///
    /// The main reason we need to store this (rather than calculate it on the fly via `list_size`
    /// and `values.len()`) is because in the degenerate case where `list_size == 0`, we cannot use
    /// `0 / 0` to determine the length.
    len: usize,

    /// The stats for this array.
    stats_set: ArrayStats,
}

impl FixedSizeListArray {
    pub fn new(values: ArrayRef, list_size: u32, validity: Validity, len: usize) -> Self {
        Self::try_new(values, list_size, validity, len)
            .vortex_expect("FixedSizeListArray `try_new` failed")
    }

    pub fn try_new(
        values: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<Self> {
        let nullability = validity.nullability();

        Self::validate(&values, len, list_size, &validity)?;

        Ok(Self {
            dtype: DType::FixedSizeList(Arc::new(values.dtype().clone()), list_size, nullability),
            values,
            list_size,
            validity,
            len,
            stats_set: Default::default(),
        })
    }

    /// Returns the values array.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// The size of each fixed-size list scalar in the array.
    pub const fn list_size(&self) -> u32 {
        self.list_size
    }

    // TODO(connor)[FixedSizeList]: Don't we need to take the validity into consideration here?
    /// Returns the elements at the given index from the list array.
    pub fn fixed_size_list_at(&self, index: usize) -> ArrayRef {
        let start = self.list_size as usize * index;
        let end = self.list_size as usize * (index + 1);
        self.values().slice(start, end)
    }

    /// Checks if the components of a `FixedSizeListArray` are valid.
    ///
    /// A fixed-size list array is valid if:
    ///
    /// - The `list_size` is 0 and:
    ///   - The `values` array is empty.
    ///   - The `len` is equal to the length of the `validity` map.
    /// - The length of the `values` array is a multiple of the size of the fixed-size lists
    ///   (`list_size`).
    /// - The `Validity` length (if it exists) times the `list_size` is equal to the length of the
    ///   `values` (or put another way, the length of the array divided by the size of each
    ///   fixed-size list is equal to the length of the validity).
    fn validate(
        values: &dyn Array,
        len: usize,
        list_size: u32,
        validity: &Validity,
    ) -> VortexResult<()> {
        // A fixed-size list array where each list scalar is empty is completely useless, but we can
        // support it regardless.
        if list_size == 0 {
            vortex_ensure!(
                values.is_empty() && validity.maybe_len().is_none_or(|vlen| vlen == len),
                "an empty `FixedSizeList` should have no values"
            );
            return Ok(());
        }

        let num_values = values.len();

        vortex_ensure!(
            len * list_size as usize == num_values,
            "the `values` array has the incorrect number of values to construct a \
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
