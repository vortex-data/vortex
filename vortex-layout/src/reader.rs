use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::ArrayData;
use vortex_dtype::{DType, FieldPath};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_scan::RowMask;

use crate::LayoutData;

/// A [`LayoutReader`] is an instance of a [`LayoutData`] that can cache state across multiple
/// operations.
///
/// Since different row ranges of the reader may be evaluated by different threads, it is required
/// to be both `Send` and `Sync`.
pub trait LayoutReader: Send + Sync + ExprEvaluator + StatsEvaluator {
    /// Returns the [`LayoutData`] of this reader.
    fn layout(&self) -> &LayoutData;
}

impl LayoutReader for Arc<dyn LayoutReader + 'static> {
    fn layout(&self) -> &LayoutData {
        self.as_ref().layout()
    }
}

/// A trait for evaluating expressions against a [`LayoutReader`].
#[async_trait(?Send)]
pub trait ExprEvaluator {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData>;
}

#[async_trait(?Send)]
impl ExprEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData> {
        self.as_ref().evaluate_expr(row_mask, expr).await
    }
}

/// A trait for evaluating field statistics against a [`LayoutReader`].
///
/// Implementations should avoid fetching data segments (metadata segments are ok) and instead
/// rely on the statistics that were computed at write time.
#[async_trait(?Send)]
pub trait StatsEvaluator {
    async fn evaluate_stats(
        &self,
        field_paths: &[FieldPath],
        stats: &[Stat],
    ) -> VortexResult<Vec<StatsSet>>;
}

#[async_trait(?Send)]
impl StatsEvaluator for Arc<dyn LayoutReader + 'static> {
    async fn evaluate_stats(
        &self,
        field_paths: &[FieldPath],
        stats: &[Stat],
    ) -> VortexResult<Vec<StatsSet>> {
        self.as_ref().evaluate_stats(field_paths, stats).await
    }
}

pub trait LayoutScanExt: LayoutReader {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutReader>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }

    /// Returns the DType of the layout.
    fn dtype(&self) -> &DType {
        self.layout().dtype()
    }
}

impl<L: LayoutReader> LayoutScanExt for L {}
