// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use std::sync::Arc;

use vortex::array::iter::ArrayIterator;

use crate::array::vx_array;
use crate::box_dyn_wrapper;
use crate::error::try_or_default;
use crate::error::vx_error;

box_dyn_wrapper!(
    /// A Vortex array iterator.
    ///
    /// Once the iterator is finished (returns `null` from [`vx_array_iterator_next`]), it may panic
    /// on subsequent calls to [`vx_array_iterator_next`].
    ///
    /// Even after the iterator is finished, an owned iterator must be released by calling
    /// [`vx_array_iter_free`].
    ///
    /// Iterators may be passed between threads, but calls to [`vx_array_iterator_next`] should be
    /// serialized and not invoked concurrently.
    dyn ArrayIterator,
    vx_array_iterator
);

/// Attempt to advance the `current` pointer of the iterator.
///
/// A return value of `true` indicates that another element was pulled from the iterator, and a return
/// of `false` indicates that the iterator is finished.
///
/// It is an error to call this function again after the iterator is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_iterator_next(
    iter: *mut vx_array_iterator,
    error_out: *mut *mut vx_error,
) -> *const vx_array {
    let iter = vx_array_iterator::as_mut(iter);
    try_or_default(error_out, || {
        let element = iter.next();

        if let Some(element) = element {
            Ok(vx_array::new(Arc::new(element?)))
        } else {
            // Drop the iter pointer.
            Ok(ptr::null_mut())
        }
    })
}
