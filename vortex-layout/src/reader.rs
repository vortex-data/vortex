use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::Array;
use vortex_dtype::{DType, FieldPath};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_scan::RowMask;

use crate::Layout;

/// A [`LayoutReader`] is an instance of a [`Layout`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: Send + Sync + ExprEvaluator + StatsEvaluator {
    /// Returns the [`Layout`] of this reader.
    fn layout(&self) -> &Layout;
}

impl LayoutReader for Arc<dyn LayoutReader + 'static> {
    fn layout(&self) -> &Layout {
        self.as_ref().layout()
    }
}

/// A trait for evaluating expressions against a [`LayoutReader`].
#[async_trait]
pub trait ExprEvaluator {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array>;
}

#[async_trait]
impl ExprEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        self.as_ref().evaluate_expr(row_mask, expr).await
    }
}

/// A trait for evaluating field statistics against a [`LayoutReader`].
///
/// Implementations should avoid fetching data segments (metadata segments are ok) and instead
/// rely on the statistics that were computed at write time.
#[async_trait]
pub trait StatsEvaluator {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>>;
}

#[async_trait]
impl StatsEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        self.as_ref().evaluate_stats(field_paths, stats).await
    }
}

pub trait LayoutReaderExt: LayoutReader {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutReader>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }

    /// Returns the row count of the layout.
    fn row_count(&self) -> u64 {
        self.layout().row_count()
    }

    /// Returns the DType of the layout.
    fn dtype(&self) -> &DType {
        self.layout().dtype()
    }
}

impl<L: LayoutReader> LayoutReaderExt for L {}
