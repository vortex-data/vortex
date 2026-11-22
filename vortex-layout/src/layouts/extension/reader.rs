// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delegates classic reads to the storage reader and reconstructs extension values on demand.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::ExtensionLayout;
use super::expr::rebind_to_storage;
use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::segments::SegmentSource;

/// Reader for [`ExtensionLayout`].
pub(super) struct ExtensionReader {
    layout: ExtensionLayout,
    name: Arc<str>,
    session: VortexSession,
    storage: LayoutReaderRef,
    storage_root: BoundExpression,
    /// Reuse the rebound tree across splits because the storage reader caches by tree identity.
    rebound: DashMap<ExactBoundExpr, Option<BoundExpression>>,
}

impl ExtensionReader {
    pub(super) fn try_new(
        layout: ExtensionLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<Self> {
        let storage = layout.storage()?.new_reader(
            format!("{name}.storage").into(),
            segment_source,
            session,
            ctx,
        )?;
        let storage_root = BoundExpression::new_root(layout.ext_dtype().storage_dtype().clone());

        Ok(Self {
            layout,
            name,
            session: session.clone(),
            storage,
            storage_root,
            rebound: DashMap::with_hasher_and_shard_amount(Default::default(), 8),
        })
    }

    /// Returns `expr` rebound to the storage, or `None` when it reads the extension values.
    fn storage_expr(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if expr.is_root() {
            return Ok(None);
        }

        let key = ExactBoundExpr(expr.clone());
        if let Some(rebound) = self.rebound.get(&key) {
            return Ok(rebound.value().clone());
        }

        let rebound = rebind_to_storage(expr, self.storage_root.dtype())?;
        Ok(self.rebound.entry(key).or_insert(rebound).value().clone())
    }

    /// Reads the storage of the rows `mask` selects and wraps it back into extension values.
    fn project_values(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let storage = self
            .storage
            .projection_evaluation(row_range, &self.storage_root, mask)?;
        let ext_dtype = self.layout.ext_dtype().clone();

        Ok(async move {
            let storage = storage.await?;
            Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
        }
        .boxed())
    }
}

impl LayoutReader for ExtensionReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn can_prune_file(&self, expr: &BoundExpression) -> VortexResult<bool> {
        match self.storage_expr(expr)? {
            Some(storage_expr) => self.storage.can_prune_file(&storage_expr),
            None => Ok(false),
        }
    }

    fn split_pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        match self.storage_expr(expr)? {
            Some(storage_expr) => {
                self.storage
                    .split_pruning_evaluation(row_range, &storage_expr, mask)
            }
            None => Ok(MaskFuture::ready(mask)),
        }
    }

    fn split_pruning_is_statistics(&self, expr: &BoundExpression) -> VortexResult<bool> {
        match self.storage_expr(expr)? {
            Some(storage_expr) => self.storage.split_pruning_is_statistics(&storage_expr),
            None => Ok(true),
        }
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Field paths stop at the extension, so a mask for the extension root also applies
        // to the storage root.
        self.storage
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        match self.storage_expr(expr)? {
            Some(storage_expr) => self
                .storage
                .pruning_evaluation(row_range, &storage_expr, mask),
            None => Ok(MaskFuture::ready(mask)),
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if let Some(storage_expr) = self.storage_expr(expr)? {
            return self
                .storage
                .filter_evaluation(row_range, &storage_expr, mask);
        }

        let len = mask.len();
        let input = mask.clone();
        let values = self.project_values(row_range, mask)?;
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(len, async move {
            let (input, values) = try_join!(input, values)?;
            let predicate = values
                .apply_bound(&expr)?
                .fill_null(false)?
                .execute::<Mask>(&mut session.create_execution_ctx())?;
            Ok(input.intersect_by_rank(&predicate))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        if let Some(storage_expr) = self.storage_expr(expr)? {
            return self
                .storage
                .projection_evaluation(row_range, &storage_expr, mask);
        }

        let values = self.project_values(row_range, mask)?;
        if expr.is_root() {
            return Ok(values);
        }

        let expr = expr.clone();
        Ok(async move { values.await?.apply_bound(&expr) }.boxed())
    }
}
