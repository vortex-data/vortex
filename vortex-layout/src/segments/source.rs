// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Signal that a logical batch of requests has been fully registered.
    ///
    /// Implementations that perform I/O coalescing can use this hint to promote
    /// all pending requests, allowing the coalescer to form optimal reads over
    /// the entire batch.
    fn flush(&self) {}
}

thread_local! {
    static REQUEST_COUNT_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` while counting logical segment requests issued through the top-level segment source.
pub fn with_request_count_scope<T>(f: impl FnOnce() -> T) -> (T, usize) {
    REQUEST_COUNT_STACK.with(|stack| stack.borrow_mut().push(0));
    let result = f();
    let count = REQUEST_COUNT_STACK
        .with(|stack| stack.borrow_mut().pop())
        .unwrap_or_default();
    (result, count)
}

pub(crate) fn record_segment_request() {
    REQUEST_COUNT_STACK.with(|stack| {
        if let Some(count) = stack.borrow_mut().last_mut() {
            *count = count.saturating_add(1);
        }
    });
}

/// Record multiple segment requests at once. Useful for propagating nested scope counts
/// back to the parent scope.
pub(crate) fn record_segment_requests(n: usize) {
    REQUEST_COUNT_STACK.with(|stack| {
        if let Some(count) = stack.borrow_mut().last_mut() {
            *count = count.saturating_add(n);
        }
    });
}
