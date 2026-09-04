// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Default filling for row-loop output storage.
//!
//! [`FillDefault`] lets the default [`OutputSink::initialize_skipped_rows`] zero-initialize a
//! sink's rows without knowing their representation. [`Preinitialized`] satisfies the same bound
//! with a no-op for storage that is fully initialized at construction.
//!
//! [`OutputSink::initialize_skipped_rows`]: crate::scalar_fn::unstable::row::OutputSink::initialize_skipped_rows

use super::ViewLen;

/// Output row storage that can fill itself with default placeholder values.
///
/// [`OutputSink::Rows`] requires this so the default
/// [`OutputSink::initialize_skipped_rows`] can make skipped rows safe to finish. The written
/// values are placeholders only: valid rows are overwritten by the kernel, and batch execution
/// masks skipped rows before the output is observable.
///
/// The blanket slice implementation writes `T::default()` into every element. Coherence with that
/// blanket prevents an implementation for raw `MaybeUninit` slices, so a sink exposing
/// uninitialized storage implements this on its own row view type, like
/// [`UninitElementSink`](crate::scalar_fn::unstable::row::UninitElementSink).
///
/// [`OutputSink::Rows`]: crate::scalar_fn::unstable::row::OutputSink::Rows
/// [`OutputSink::initialize_skipped_rows`]: crate::scalar_fn::unstable::row::OutputSink::initialize_skipped_rows
pub trait FillDefault {
    /// Fill every element with its default value.
    fn fill_default(&mut self);
}

impl FillDefault for () {
    fn fill_default(&mut self) {}
}

impl<T: FillDefault + ?Sized> FillDefault for &mut T {
    fn fill_default(&mut self) {
        T::fill_default(self)
    }
}

impl<T: Default + Clone> FillDefault for [T] {
    fn fill_default(&mut self) {
        self.fill(T::default());
    }
}

impl<T: Default + Clone> FillDefault for Vec<T> {
    fn fill_default(&mut self) {
        self.as_mut_slice().fill_default();
    }
}

/// Row storage whose construction already initialized every row.
///
/// A sink whose `with_capacity` returns fully initialized rows can wrap its row view in this type.
/// It satisfies the [`FillDefault`] bound on [`OutputSink::Rows`] with a no-op, so the default
/// [`OutputSink::initialize_skipped_rows`] skips redundant filling, and element types without a
/// [`Default`] implementation need none.
///
/// [`OutputSink::Rows`]: crate::scalar_fn::unstable::row::OutputSink::Rows
/// [`OutputSink::initialize_skipped_rows`]: crate::scalar_fn::unstable::row::OutputSink::initialize_skipped_rows
pub struct Preinitialized<R>(pub R);

impl<R: ViewLen> ViewLen for Preinitialized<R> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<R> FillDefault for Preinitialized<R> {
    fn fill_default(&mut self) {}
}
