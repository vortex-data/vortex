// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical optimizer rules that use Vortex file statistics.

mod cardinality;
mod join_reorder;

use std::sync::Arc;

use datafusion_physical_optimizer::PhysicalOptimizerRule;
pub use join_reorder::JoinReorder;

/// Insert [`JoinReorder`] into `rules` just before DataFusion's `JoinSelection`, which then picks
/// each reordered join's build side and partition mode. Appends it if `JoinSelection` is absent.
///
/// `JoinSelection` runs before `FilterPushdown`, so the reorder sees each base relation's filters
/// as `FilterExec` nodes and can estimate their selectivity.
pub fn insert_join_reorder(rules: &mut Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>) {
    let position = rules
        .iter()
        .position(|rule| rule.name() == "join_selection")
        .unwrap_or(rules.len());
    rules.insert(position, Arc::new(JoinReorder::new()));
}

#[cfg(test)]
mod tests;
