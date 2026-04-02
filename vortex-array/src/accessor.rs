// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Trait for arrays that support iterative access to their elements.
pub trait ArrayAccessor<Item: ?Sized> {
    /// Iterate over each element of the array, in-order.
    ///
    /// The function `f` will be passed an [`Iterator`], it can call [`Iterator::next`] on the
    /// iterator `len` times. Iterator elements are `Option` types,
    /// regardless of the nullability of the underlying array data.
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a Item>>) -> R;
}
