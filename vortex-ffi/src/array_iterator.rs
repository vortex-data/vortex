use std::ptr;

use vortex::error::{VortexExpect, vortex_bail};
use vortex::iter::ArrayIterator;

use crate::array::vx_array;
use crate::box_wrapper;
use crate::error::{try_or, vx_error};

box_wrapper!(
    /// A Vortex array iterator.
    ///
    /// Once the iterator is finished (returns `null` from [`vx_array_iter_next`]), it may panic
    /// on subsequent calls to [`vx_array_iter_next`].
    ///
    /// Even after the iterator is finished, an owned iterator must be released by calling
    /// [`vx_array_iter_free`].
    ///
    /// Iterators may be passed between threads, but calls to [`vx_array_iter_next`] should be
    /// serialized and not invoked concurrently.
    dyn ArrayIterator,
    vx_array_iterator
);

/// The FFI interface for an [`ArrayIterator`].
#[allow(non_camel_case_types)]
pub struct vx_array_iterator {
    pub inner: Option<Box<dyn ArrayIterator>>,
}

/// Attempt to advance the `current` pointer of the iterator.
///
/// A return value of `true` indicates that another element was pulled from the iterator, and a return
/// of `false` indicates that the iterator is finished.
///
/// It is an error to call this function again after the iterator is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_iter_next(
    iter: *mut vx_array_iterator,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or(error, ptr::null_mut(), || {
        let iter = unsafe { iter.as_mut() }.vortex_expect("iter null");
        let Some(inner) = iter.inner.as_mut() else {
            vortex_bail!("vx_array_iter_next called after finish")
        };

        let element = inner.next();

        if let Some(element) = element {
            Ok(vx_array::new(element?))
        } else {
            // Drop the iter pointer.
            iter.inner.take();
            Ok(ptr::null_mut())
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_iter_free(array_iter: *mut vx_array_iterator) {
    assert!(!array_iter.is_null());
    let iter = unsafe { Box::from_raw(array_iter) };
    assert!(
        iter.inner.is_none(),
        "vx_array_iter_free called before finish"
    );
    drop(iter);
}
