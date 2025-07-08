// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_array::stats::{Precision, Stat, StatsProviderExt, StatsSet};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{AccessPath, ExprRef, Identifier, Scope, ScopeDType, StatsCatalog, lit};
use vortex_layout::{ArrayEvaluation, LayoutReader, MaskEvaluation, PruningEvaluation};
use vortex_mask::Mask;

/// A [`LayoutReader`] that can prune the entire file based on file-level statistics.
pub struct FileStatsLayoutReader {
    name: Arc<str>,
    child: Arc<dyn LayoutReader>,

    /// File stats as fetched from [`crate::VortexFile`].
    file_stats: Arc<[StatsSet]>,
}

impl FileStatsLayoutReader {
    pub(crate) fn new(
        name: Arc<str>,
        child: Arc<dyn LayoutReader>,
        file_stats: Arc<[StatsSet]>,
    ) -> Self {
        Self {
            name,
            child,
            file_stats,
        }
    }
}

impl LayoutReader for FileStatsLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn scope_dtype(&self) -> &ScopeDType {
        self.child.scope_dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // We don't even wait for a mask, we just evaluate the pruning expression now.
        if let Some(falsy) = expr.stat_falsification(self) {
            if falsy
                .evaluate(&Scope::empty(1))?
                .scalar_at(0)?
                .as_bool()
                .value()
                .vortex_expect("falsy expression must evaluate to a boolean")
            {
                return Ok(Box::new(AllFalsePruningEvaluation));
            }
        }

        self.child.pruning_evaluation(row_range, expr)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        // TODO(ngates): we could attempt to perform some constant folding here based on stats.
        self.child.filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // TODO(ngates): we could attempt to perform some constant folding here based on stats.
        self.child.projection_evaluation(row_range, expr)
    }
}

/// A stats catalog that provides access to the literal values of file-level statistics
impl StatsCatalog for FileStatsLayoutReader {
    fn stats_ref(&self, access_path: &AccessPath, stat: Stat) -> Option<ExprRef> {
        if access_path.identifier() != &Identifier::Identity {
            return None;
        }

        // NOTE(ngates): for now, file stats are only available for root columns. So our access
        //  path must have length = 1.
        let path = access_path.field_path().path();
        println!("STATS PATH: {path:?}");
        if path.len() != 1 {
            return None;
        }

        let struct_fields = self.dtype().as_struct()?;
        let field_idx = struct_fields.find(path[0].as_name()?)?;
        let field_dtype = struct_fields.field_by_index(field_idx)?;

        let value = self
            .file_stats
            .get(field_idx)?
            .get_scalar(stat, &field_dtype)?
            .as_exact()?;

        Some(lit(value))
    }
}

struct AllFalsePruningEvaluation;

#[async_trait]
impl PruningEvaluation for AllFalsePruningEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        Ok(Mask::new_false(mask.len()))
    }
}
