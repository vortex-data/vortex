// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations for filtering data using a [`BitView`] selection mask.
//!
//! These implementations are highly optimized as filtering is such a core operation within Vortex.

pub(crate) mod scalar_in_place;

use crate::BitView;

impl<'a, const NB: usize> BitView<'a, NB> {
    /// Filters the given slice of items in place.
    ///
    /// After calling this method, the first `self.true_count()` elements of `items`
    /// will contain the filtered items. The remaining elements beyond that point are undefined.
    pub fn filter_in_place<T: Copy>(&self, items: &mut [T]) {
        match self.true_count() {
            0 => {
                // No items to keep; do nothing.
            }
            n if n == items.len() => {
                // All items to keep; do nothing.
            }
            _ => {
                // Some items to keep; do the filtering.
                scalar_in_place::filter_in_place_scalar(self, items);
            }
        }
    }
}
