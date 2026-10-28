// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::Weak;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::FilePruner;
use parking_lot::Mutex;
use vortex_utils::aliases::dash_map::DashMap;

use super::source::FileVersion;

#[derive(PartialEq, Eq, Hash)]
struct PruningKey {
    file: FileVersion,
    predicate: usize,
    schema: usize,
    statistics: usize,
}

/// Shares pruning work only while partitions retain the same file and predicate state.
#[derive(Default)]
pub(crate) struct FilePruners {
    entries: DashMap<PruningKey, Weak<Mutex<SharedPruning>>>,
}

struct SharedPruning {
    pruner: FilePruner,
    pruned: bool,
}

enum PruningState {
    Owned(FilePruner),
    Shared(Arc<Mutex<SharedPruning>>),
}

pub(crate) struct Pruner(PruningState);

impl Pruner {
    pub fn try_new(
        shared: Option<&FilePruners>,
        predicate: PhysicalExprRef,
        schema: &SchemaRef,
        file: &PartitionedFile,
        errors: Count,
    ) -> Option<Self> {
        let Some(shared) = shared else {
            return FilePruner::try_new(predicate, schema, file, errors)
                .map(|pruner| Self(PruningState::Owned(pruner)));
        };
        let statistics = file.statistics.as_ref()?;
        let key = PruningKey {
            file: FileVersion(file.object_meta.clone()),
            predicate: Arc::as_ptr(&predicate) as *const () as usize,
            schema: Arc::as_ptr(schema) as usize,
            statistics: Arc::as_ptr(statistics) as usize,
        };
        if shared.entries.len() > 1024 {
            shared.entries.retain(|_, value| value.strong_count() != 0);
        }
        let mut entry = shared.entries.entry(key).or_default();
        if let Some(pruner) = entry.upgrade() {
            return Some(Self(PruningState::Shared(pruner)));
        }
        // The pruner retains these identities while the weak entry can be upgraded.
        // Rewritten partition literals have distinct predicate identities and cannot alias.
        let pruner = FilePruner::try_new(predicate, schema, file, errors)?;
        let shared = Arc::new(Mutex::new(SharedPruning {
            pruner,
            pruned: false,
        }));
        *entry = Arc::downgrade(&shared);
        Some(Self(PruningState::Shared(shared)))
    }

    pub fn should_prune(&mut self) -> DFResult<bool> {
        match &mut self.0 {
            PruningState::Owned(pruner) => pruner.should_prune(),
            PruningState::Shared(shared) => {
                let mut state = shared.lock();
                // FilePruner reports changes, so an unchanged check after a successful
                // prune returns false. Other partitions must retain the terminal decision.
                if !state.pruned {
                    state.pruned = state.pruner.should_prune()?;
                }
                Ok(state.pruned)
            }
        }
    }

    pub fn is_watching(&self) -> bool {
        match &self.0 {
            PruningState::Owned(pruner) => pruner.is_watching(),
            PruningState::Shared(shared) => shared.lock().pruner.is_watching(),
        }
    }
}

#[cfg(test)]
mod tests;
