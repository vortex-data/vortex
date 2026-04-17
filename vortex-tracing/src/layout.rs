// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use tracing::Instrument;
use tracing::field;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_mask::Mask;

use crate::TARGET_LAYOUT;

/// A decorator that emits [`tracing`] spans around every evaluation call on
/// the wrapped [`LayoutReader`].
///
/// Each span records the layout name, row range, and expression being
/// evaluated. Spans are opened for `pruning_evaluation`, `filter_evaluation`,
/// `projection_evaluation`, and `register_splits` and follow the returned
/// future across `.await` points.
///
/// Wrap the root layout reader before passing it to `ScanBuilder::new` to
/// capture evaluation on the root. To trace child layouts, construct a new
/// [`TracingLayoutReader`] around each child reader when it is produced (see
/// the crate-level docs for the recommended injection point).
pub struct TracingLayoutReader<R: ?Sized> {
    inner: Arc<R>,
}

impl<R: LayoutReader + ?Sized> TracingLayoutReader<R> {
    /// Wrap an existing [`LayoutReader`] so that its evaluation calls are
    /// traced.
    pub fn new(inner: Arc<R>) -> Self {
        Self { inner }
    }
}

impl TracingLayoutReader<dyn LayoutReader> {
    /// Convenience constructor for the common case of wrapping an already
    /// erased [`Arc<dyn LayoutReader>`].
    pub fn wrap(inner: Arc<dyn LayoutReader>) -> Arc<dyn LayoutReader> {
        Arc::new(Self { inner })
    }
}

impl<R: LayoutReader + ?Sized> LayoutReader for TracingLayoutReader<R> {
    fn name(&self) -> &Arc<str> {
        self.inner.name()
    }

    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let _guard = tracing::info_span!(
            target: TARGET_LAYOUT,
            "register_splits",
            layout = %self.inner.name(),
            row_start = row_range.start,
            row_end = row_range.end,
            field_mask_len = field_mask.len(),
        )
        .entered();
        self.inner.register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let span = tracing::info_span!(
            target: TARGET_LAYOUT,
            "pruning_evaluation",
            layout = %self.inner.name(),
            row_start = row_range.start,
            row_end = row_range.end,
            expr = %expr,
            duration_us = field::Empty,
        );
        let fut = self.inner.pruning_evaluation(row_range, expr, mask)?;
        let len = fut.len();
        Ok(MaskFuture::new(len, timed(fut, span)))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let span = tracing::info_span!(
            target: TARGET_LAYOUT,
            "filter_evaluation",
            layout = %self.inner.name(),
            row_start = row_range.start,
            row_end = row_range.end,
            expr = %expr,
            input_mask_len = mask.len(),
            duration_us = field::Empty,
        );
        let fut = self.inner.filter_evaluation(row_range, expr, mask)?;
        let len = fut.len();
        Ok(MaskFuture::new(len, timed(fut, span)))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let span = tracing::info_span!(
            target: TARGET_LAYOUT,
            "projection_evaluation",
            layout = %self.inner.name(),
            row_start = row_range.start,
            row_end = row_range.end,
            expr = %expr,
            input_mask_len = mask.len(),
            duration_us = field::Empty,
        );
        let fut = self.inner.projection_evaluation(row_range, expr, mask)?;
        Ok(Box::pin(timed(fut, span)))
    }
}

async fn timed<F, T>(fut: F, span: tracing::Span) -> T
where
    F: Future<Output = T>,
{
    async move {
        let start = std::time::Instant::now();
        let out = fut.await;
        tracing::Span::current().record(
            "duration_us",
            u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
        out
    }
    .instrument(span)
    .await
}
