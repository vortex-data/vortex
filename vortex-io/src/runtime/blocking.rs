// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::Stream;

use crate::runtime::Handle;

/// A generic API blocking entry points to runtimes.
pub trait BlockingRuntime {
    /// Associated type for the blocking iterator returned by `block_on_stream`.
    type BlockingIterator<'a, R>: Iterator<Item = R> + 'a
    where
        R: 'a;

    /// Runs a future to completion on the runtime, blocking the current thread until it completes.
    ///
    /// The future is provided a [`Handle`] to the runtime so that it may spawn additional tasks
    /// to be executed concurrently.
    fn block_on<F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>;

    /// Returns an iterator wrapper around a stream, blocking the current thread for each item.
    fn block_on_stream<'a, F, S, R>(&self, f: F) -> Self::BlockingIterator<'a, R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + Unpin + 'a,
        R: Send + 'a;
}
