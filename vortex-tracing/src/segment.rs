// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use tracing::Instrument;
use tracing::field;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;

use crate::TARGET_SEGMENT;

/// A decorator that emits a [`tracing`] span around every logical segment
/// request to the wrapped [`SegmentSource`].
///
/// Each `request` call produces a span named `"segment_request"` with field
/// `segment_id` and a `duration_us` recorded on completion. This is the
/// pre-coalescing view of I/O: when combined with spans from
/// [`crate::TracingReadAt`], you can identify which logical segment requests
/// were merged into each physical read by post-processing on byte-range
/// overlap.
pub struct TracingSegmentSource<S> {
    inner: S,
}

impl<S> TracingSegmentSource<S> {
    /// Wrap an existing [`SegmentSource`] so that each segment request is
    /// traced.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Returns a reference to the wrapped segment source.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S> TracingSegmentSource<S>
where
    S: SegmentSource,
{
    /// Wrap and return as an [`Arc<dyn SegmentSource>`] ready for use at a
    /// point that expects the erased trait object.
    pub fn into_arc(self) -> Arc<dyn SegmentSource> {
        Arc::new(self)
    }
}

impl<S: SegmentSource> SegmentSource for TracingSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let span = tracing::info_span!(
            target: TARGET_SEGMENT,
            "segment_request",
            segment_id = *id,
            duration_us = field::Empty,
        );
        let inner = self.inner.request(id);
        Box::pin(
            async move {
                let start = std::time::Instant::now();
                let result = inner.await;
                tracing::Span::current().record(
                    "duration_us",
                    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
                );
                result
            }
            .instrument(span),
        )
    }
}
