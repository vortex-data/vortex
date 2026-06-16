// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex array iterators.
//!
//! In the hand-written C ABI this was a `box_dyn_wrapper!(dyn ArrayIterator, vx_array_iterator)`
//! plus a single `vx_array_iterator_next` function that returned the next array (or null at the
//! end) and reported errors through an `error_out` out-parameter, and the macro-generated
//! `vx_array_iterator_free`.
//!
//! Under Diplomat we model this with Diplomat's first-class iterator support. The opaque
//! [`VxArrayIterator`] exposes a `next` method tagged `#[diplomat::attr(auto, iterator)]` that
//! returns `Option<Box<VxArray>>` — `None` signals exhaustion, exactly like the C ABI returning a
//! null pointer. Diplomat renders this as a native iterator (`for`-loop / `Iterator` / generator)
//! in each target language, and auto-generates the destructor.
//!
//! Note: the C ABI surfaced per-element decode errors through `error_out`. Diplomat iterators
//! cannot yield a `Result` per element, so a decode error here is converted into early
//! termination (`None`); fallible iteration should be performed by collecting via a method that
//! returns `Result` if strict error propagation is required.

pub use ffi::VxArrayIterator;

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use vortex::array::iter::ArrayIterator;

    use crate::array::ffi::VxArray;

    /// A Vortex array iterator.
    ///
    /// Yields owned [`VxArray`] handles until exhausted. Iterators may be moved between threads,
    /// but `next` must not be called concurrently on the same iterator.
    #[diplomat::opaque]
    pub struct VxArrayIterator(pub(crate) Box<dyn ArrayIterator>);

    impl VxArrayIterator {
        /// Advance the iterator, returning the next array or `None` when finished.
        ///
        /// Replaces `vx_array_iterator_next`. A return of `None` is the Diplomat analogue of the C
        /// ABI returning a null pointer to signal the end of iteration. Each yielded `VxArray` is
        /// owned by the caller; Diplomat generates the destructor.
        #[diplomat::attr(auto, iterator)]
        pub fn next(&mut self) -> Option<Box<VxArray>> {
            match self.0.next()? {
                Ok(array) => Some(Box::new(VxArray(Arc::new(array)))),
                // A per-element decode error terminates iteration; see module docs.
                Err(_) => None,
            }
        }
    }
}
