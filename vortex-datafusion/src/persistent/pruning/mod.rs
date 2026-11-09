// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;
use std::sync::OnceLock;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::pruning::PrunableStatistics;
use datafusion_datasource::PartitionedFile;
use datafusion_expr::Operator;
use datafusion_expr::Volatility;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::CaseExpr;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::expressions::InListExpr;
use datafusion_physical_expr::expressions::IsNotNullExpr;
use datafusion_physical_expr::expressions::IsNullExpr;
use datafusion_physical_expr::expressions::LikeExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::NegativeExpr;
use datafusion_physical_expr::expressions::NotExpr;
use datafusion_physical_expr::expressions::TryCastExpr;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_plan::joins::HashExpr;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::PruningPredicate;
use datafusion_pruning::build_pruning_predicate;
use parking_lot::RwLock;
use tokio::sync::OnceCell;

/// One source's file-pruning expression. Compiling it for every file otherwise clones
/// and drops the same shared expression tree on every opening worker.
///
/// Replace this cache whenever the source's predicate changes. Partition literals must not
/// use it: their expression differs per file. Known dynamic predicates are cached by
/// generation; custom expressions retain DataFusion's per-file compilation behavior.
/// Each file retains its own update watcher and statistics.
/// Runtime filters must refine during execution; resetting them requires a new predicate epoch.
#[derive(Default)]
pub(super) struct FilePruningCache {
    kind: OnceLock<CacheKind>,
    predicate: OnceLock<Option<Arc<PruningPredicate>>>,
    dynamic_predicate: RwLock<Option<Arc<VersionedPredicate>>>,
}

type CompiledPredicate = Option<Arc<PruningPredicate>>;

struct VersionedPredicate {
    generation: u64,
    predicate: OnceCell<CompiledPredicate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheKind {
    Static,
    Dynamic,
    Uncached,
}

impl CacheKind {
    fn classify(expression: &PhysicalExprRef) -> Self {
        let mut kind = Self::Static;
        let mut pending = vec![(expression, true)];
        while let Some((node, positive)) = pending.pop() {
            if node.is::<DynamicFilterPhysicalExpr>() {
                if !positive {
                    return Self::Uncached;
                }
                kind = Self::Dynamic;
            } else if !is_immutable_node(node) {
                tracing::trace!(expression = %node, "Bypassing shared pruning for opaque expression");
                return Self::Uncached;
            }
            // Only positive Boolean composition preserves the filter's refinement order.
            let positive_children = positive
                && node
                    .downcast_ref::<BinaryExpr>()
                    .is_some_and(|binary| matches!(binary.op(), Operator::And | Operator::Or));
            pending.extend(
                node.children()
                    .into_iter()
                    .map(|child| (child, positive_children)),
            );
        }
        kind
    }
}

fn is_immutable_node(expression: &PhysicalExprRef) -> bool {
    if let Some(function) = expression.downcast_ref::<ScalarFunctionExpr>() {
        // Stable functions can change between executions of the same source.
        return function.fun().signature().volatility == Volatility::Immutable;
    }
    // Unknown implementations can expose mutable snapshots even at generation zero.
    // These nodes have fixed state and expose every expression input through their children.
    expression.is::<Column>()
        || expression.is::<Literal>()
        || expression.is::<HashExpr>()
        || expression.is::<BinaryExpr>()
        || expression.is::<CastExpr>()
        || expression.is::<TryCastExpr>()
        || expression.is::<NotExpr>()
        || expression.is::<NegativeExpr>()
        || expression.is::<IsNullExpr>()
        || expression.is::<IsNotNullExpr>()
        || expression.is::<LikeExpr>()
        || expression.is::<InListExpr>()
        || expression.is::<CaseExpr>()
}

impl FilePruningCache {
    fn kind(&self, expression: &PhysicalExprRef) -> CacheKind {
        *self.kind.get_or_init(|| CacheKind::classify(expression))
    }

    /// `None` delegates dynamic predicates to FilePruner; `Some` is this file's static result.
    pub(super) fn should_prune(
        &self,
        expression: &PhysicalExprRef,
        schema: &SchemaRef,
        file: &PartitionedFile,
        errors: &Count,
    ) -> Option<bool> {
        if self.kind(expression) != CacheKind::Static {
            return None;
        }
        // A file without usable statistics must not prevent a later file from compiling the
        // shared predicate. This matches FilePruner's static-predicate eligibility check.
        if !file.has_statistics() {
            return Some(false);
        }
        let statistics = file.statistics.as_ref()?;
        let compiled = match self.predicate.get() {
            Some(compiled) => compiled,
            None => {
                // A cold partition can compile its own predicate rather than blocking a runtime
                // worker on another partition's compilation. Later files reuse the first result.
                let compiled = build_pruning_predicate(Arc::clone(expression), schema, errors);
                self.predicate.get_or_init(|| compiled)
            }
        };
        let Some(predicate) = compiled else {
            return Some(false);
        };
        let statistics = PrunableStatistics::new(vec![Arc::clone(statistics)], Arc::clone(schema));
        Some(prune_file(predicate, &statistics, errors))
    }

    fn dynamic_predicate(
        &self,
        expression: &PhysicalExprRef,
        build: impl FnOnce() -> CompiledPredicate,
    ) -> CompiledPredicate {
        let generation = snapshot_generation(expression);
        let entry = self.dynamic_entry(generation);
        // Within this predicate epoch, an older filter admits a superset of the current
        // matches. A stale snapshot can miss pruning, but cannot discard a matching file.
        if let Some(predicate) = entry.predicate.get() {
            return predicate.clone();
        }
        // Stream polling cannot await another initializer. A competing build remains local.
        let predicate = build();
        if snapshot_generation(expression) == generation {
            drop(entry.predicate.set(predicate.clone()));
        }
        predicate
    }

    async fn initial_dynamic_predicate<Fut>(
        &self,
        expression: &PhysicalExprRef,
        build: impl FnOnce() -> Fut,
    ) -> CompiledPredicate
    where
        Fut: Future<Output = CompiledPredicate>,
    {
        let generation = snapshot_generation(expression);
        let entry = self.dynamic_entry(generation);
        match entry
            .predicate
            .get_or_try_init(|| async {
                // A waiter on an obsolete cell can skip this optional check. Its file watcher
                // still holds the update, so stream polling will check the new generation.
                if snapshot_generation(expression) != generation {
                    return Err(None);
                }
                let predicate = build().await;
                if snapshot_generation(expression) == generation {
                    Ok(predicate)
                } else {
                    Err(predicate)
                }
            })
            .await
        {
            Ok(predicate) => predicate.clone(),
            Err(predicate) => predicate,
        }
    }

    fn dynamic_entry(&self, generation: u64) -> Arc<VersionedPredicate> {
        {
            let cached = self.dynamic_predicate.read();
            if let Some(cached) = cached.as_ref()
                && cached.generation == generation
            {
                return Arc::clone(cached);
            }
        }
        let entry = Arc::new(VersionedPredicate {
            generation,
            predicate: OnceCell::new(),
        });
        let previous = {
            let mut cached = self.dynamic_predicate.write();
            if let Some(cached) = cached.as_ref()
                && cached.generation == generation
            {
                return Arc::clone(cached);
            }
            if cached
                .as_ref()
                .is_none_or(|cached| cached.generation < generation)
            {
                cached.replace(Arc::clone(&entry))
            } else {
                None
            }
        };
        // Dropping an old expression tree can be expensive; do it after releasing the lock.
        drop(previous);
        entry
    }
}

pub(super) enum FilePruner {
    Static(bool),
    Shared(SharedFilePruner),
    DataFusion(datafusion_pruning::FilePruner),
}

impl FilePruner {
    pub(super) fn try_new(
        predicate: PhysicalExprRef,
        schema: &SchemaRef,
        file: &PartitionedFile,
        errors: Count,
        cache: Option<Arc<FilePruningCache>>,
    ) -> Option<Self> {
        if let Some(pruned) = cache
            .as_ref()
            .and_then(|cache| cache.should_prune(&predicate, schema, file, &errors))
        {
            return Some(Self::Static(pruned));
        }
        let cache = cache.filter(|cache| cache.kind(&predicate) != CacheKind::Uncached);
        let Some(cache) = cache else {
            return datafusion_pruning::FilePruner::try_new(predicate, schema, file, errors)
                .map(Self::DataFusion);
        };
        let statistics = file.statistics.as_ref()?;
        let tracking = DynamicFilterTracking::classify(&predicate);
        if !file.has_statistics() && !tracking.contains_dynamic_filter() {
            return None;
        }
        Some(Self::Shared(SharedFilePruner {
            predicate,
            tracking,
            checked_once: false,
            schema: Arc::clone(schema),
            statistics: PrunableStatistics::new(vec![Arc::clone(statistics)], Arc::clone(schema)),
            errors,
            cache,
        }))
    }

    pub(super) fn is_watching(&self) -> bool {
        match self {
            Self::Static(_) => false,
            Self::Shared(pruner) => matches!(pruner.tracking, DynamicFilterTracking::Watching(_)),
            Self::DataFusion(pruner) => pruner.is_watching(),
        }
    }

    pub(super) fn should_prune(&mut self) -> DFResult<bool> {
        match self {
            Self::Static(pruned) => Ok(std::mem::take(pruned)),
            Self::Shared(pruner) => Ok(pruner.should_prune()),
            Self::DataFusion(pruner) => pruner.should_prune(),
        }
    }

    pub(super) async fn should_prune_initial(&mut self) -> DFResult<bool> {
        match self {
            Self::Static(pruned) => Ok(std::mem::take(pruned)),
            Self::Shared(pruner) => Ok(pruner.should_prune_initial().await),
            Self::DataFusion(pruner) => pruner.should_prune(),
        }
    }
}

pub(super) struct SharedFilePruner {
    predicate: PhysicalExprRef,
    tracking: DynamicFilterTracking,
    checked_once: bool,
    schema: SchemaRef,
    statistics: PrunableStatistics,
    errors: Count,
    cache: Arc<FilePruningCache>,
}

impl SharedFilePruner {
    fn needs_check(&mut self) -> bool {
        if self.checked_once {
            if !self
                .tracking
                .watcher()
                .is_some_and(|watcher| watcher.changed())
            {
                return false;
            }
        } else {
            self.checked_once = true;
        }
        true
    }

    fn should_prune(&mut self) -> bool {
        if !self.needs_check() {
            return false;
        }
        self.cache
            .dynamic_predicate(&self.predicate, || {
                build_pruning_predicate(Arc::clone(&self.predicate), &self.schema, &self.errors)
            })
            .is_some_and(|predicate| prune_file(&predicate, &self.statistics, &self.errors))
    }

    async fn should_prune_initial(&mut self) -> bool {
        if !self.needs_check() {
            return false;
        }
        self.cache
            .initial_dynamic_predicate(&self.predicate, || async {
                build_pruning_predicate(Arc::clone(&self.predicate), &self.schema, &self.errors)
            })
            .await
            .is_some_and(|predicate| prune_file(&predicate, &self.statistics, &self.errors))
    }
}

fn prune_file(
    predicate: &PruningPredicate,
    statistics: &PrunableStatistics,
    errors: &Count,
) -> bool {
    match predicate.prune(statistics) {
        Ok(keep) => matches!(keep.as_slice(), [false]),
        Err(error) => {
            tracing::debug!(%error, "Ignoring file-pruning evaluation error");
            errors.add(1);
            false
        }
    }
}

#[cfg(test)]
mod tests;
