use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::FuturesOrdered;
use futures::TryStreamExt;
use itertools::Itertools;
use vortex_array::array::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::struct_::reader::SharedState;
use crate::LayoutRangeReader;

pub struct StructRangeReader {
    /// The row range of this reader within the global file.
    pub(crate) row_range: Range<u64>,
    /// The readers for each field mentioned in field_names.
    pub(crate) fields: Vec<Arc<dyn LayoutRangeReader>>,
    /// Shared state across all field readers, e.g. pruning cache.
    pub(crate) shared_state: Arc<SharedState>,
}

#[async_trait]
impl LayoutRangeReader for StructRangeReader {
    fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.shared_state.partition_expr(expr.clone())?;

        // Spawn a task for each partition of the expression
        let mut results = FuturesOrdered::new();
        for (name, expr) in partitioned
            .partition_names
            .iter()
            .zip_eq(partitioned.partitions.iter())
        {
            let idx = self.shared_state.child_idx(name)?;
            results.push_back(self.fields[idx].evaluate_expr(mask.clone(), expr.clone()))
        }

        // Wait for all tasks to complete
        let arrays = results.into_stream().try_collect::<Vec<_>>().await?;

        let row_count = mask.true_count();
        debug_assert!(arrays.iter().all(|a| a.len() == row_count));

        let root_scope = StructArray::try_new(
            partitioned.partition_names.clone(),
            arrays,
            row_count,
            Validity::NonNullable,
        )?
        .into_array();

        partitioned.root.evaluate(&root_scope)
    }
}
