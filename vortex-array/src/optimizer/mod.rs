// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::hash_map::HashMap;

use crate::Array;
use crate::array::ArrayRef;
use crate::optimizer::rules::AnyArray;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::DynArrayParentReduceRule;
use crate::optimizer::rules::DynArrayReduceRule;
use crate::optimizer::rules::MatchKey;
use crate::optimizer::rules::Matcher;
use crate::optimizer::rules::ParentReduceRuleAdapter;
use crate::optimizer::rules::ReduceRuleAdapter;

pub mod rules;

#[cfg(test)]
mod tests;

/// Optimizer for arrays that applies registered rewrite rules.
///
/// This optimizer recursively traverses an array tree, applying reduce rules
/// to transform arrays into more efficient representations.
#[derive(Default, Debug, Clone)]
pub struct ArrayOptimizer {
    /// Reduce rules indexed by encoding ID
    reduce_rules: HashMap<MatchKey, Vec<Arc<dyn DynArrayReduceRule>>>,
    /// Parent reduce rules for specific parent types, indexed by (child, parent)
    parent_rules: HashMap<(MatchKey, MatchKey), Vec<Arc<dyn DynArrayParentReduceRule>>>,
}

impl ArrayOptimizer {
    /// Optimize only the top-level array by applying registered rewrite rules.
    ///
    /// This is useful when it is assumed that the children are already optimized, and we have
    /// simply wrapped them up in a new array, such as applying an expression.
    pub fn optimize_root(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(self.try_optimize_root(array.clone())?.unwrap_or(array))
    }

    /// Try to optimize only the top-level array by applying registered rewrite rules.
    ///
    /// Returns `Ok(None)` if no optimizations were applied, otherwise returns the optimized array.
    ///
    /// This is useful when it is assumed that the children are already optimized, and we have
    /// simply wrapped them up in a new array, such as applying an expression.
    #[allow(clippy::cognitive_complexity)]
    pub fn try_optimize_root(&self, array: ArrayRef) -> VortexResult<Option<ArrayRef>> {
        tracing::debug!(
            "Starting root-only array optimization\n{}",
            array.display_tree()
        );
        let mut current_array = array;
        let mut any_optimizations = false;

        // Apply reduction rules to the current array until no more rules apply.
        let mut loop_counter = 0;
        loop {
            if loop_counter > 100 {
                vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
            }
            loop_counter += 1;

            if let Some(new_array) = self.apply_reduce_rules(&current_array)? {
                current_array = new_array;
                any_optimizations = true;
                continue;
            }

            // Apply parent reduction rules to each child in the context of the current array.
            let mut replaced = false;
            for (idx, child) in current_array.children().iter().enumerate() {
                if let Some(new_array) = self.apply_parent_rules(child, &current_array, idx)? {
                    // If the parent was replaced, then we start over with the new parent
                    current_array = new_array;
                    any_optimizations = true;
                    replaced = true;
                    break;
                }
            }

            if !replaced {
                break;
            }
        }

        if any_optimizations {
            tracing::debug!(
                "Optimized root-only array\n{}",
                current_array.display_tree()
            );
            Ok(Some(current_array))
        } else {
            tracing::debug!("No optimizations applied to root array");
            Ok(None)
        }
    }

    /// Optimize the given array by applying registered rewrite rules recursively over all nodes.
    ///
    /// This can be quite an expensive traversal, so prefer [`ArrayOptimizer::optimize_root`]
    /// where possible, running it each time a new array is constructed.
    pub fn optimize_recursive(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(self.try_optimize_recursive(array.clone())?.unwrap_or(array))
    }

    /// Optimize the given array by applying registered rewrite rules recursively over all nodes.
    ///
    /// This can be quite an expensive traversal, so prefer [`ArrayOptimizer::optimize_root`]
    /// where possible, running it each time a new array is constructed.
    pub fn try_optimize_recursive(&self, array: ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let mut current_array = array;
        let mut any_optimizations = false;

        if let Some(new_array) = self.try_optimize_root(current_array.clone())? {
            current_array = new_array;
            any_optimizations = true;
        }

        let mut new_children = Vec::with_capacity(current_array.nchildren());
        let mut any_child_optimized = false;
        for child in current_array.children() {
            if let Some(new_child) = self.try_optimize_recursive(child.clone())? {
                new_children.push(new_child);
                any_child_optimized = true;
            } else {
                new_children.push(child.clone());
            }
        }

        if any_child_optimized {
            current_array = current_array.with_children(new_children)?;
            any_optimizations = true;
        }

        if any_optimizations {
            Ok(Some(current_array))
        } else {
            Ok(None)
        }
    }

    /// Register a reduce rule for a specific array encoding.
    pub fn register_reduce_rule<M, R>(&mut self, rule: R)
    where
        M: Matcher,
        R: ArrayReduceRule<M> + 'static,
    {
        let key = rule.matcher().key();
        let adapter = ReduceRuleAdapter::new(rule);
        self.reduce_rules
            .entry(key)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule for a specific parent type.
    pub fn register_parent_rule<Child, Parent, R>(&mut self, rule: R)
    where
        Child: Matcher,
        Parent: Matcher,
        R: ArrayParentReduceRule<Child, Parent> + 'static,
    {
        let key = (rule.child().key(), rule.parent().key());
        let adapter = ParentReduceRuleAdapter::new(rule);
        self.parent_rules
            .entry(key)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_any_parent_rule<Child, R>(&mut self, rule: R)
    where
        Child: Matcher,
        R: ArrayParentReduceRule<Child, AnyArray> + 'static,
    {
        let key = (rule.child().key(), MatchKey::Any);
        let adapter = ParentReduceRuleAdapter::new(rule);
        self.parent_rules
            .entry(key)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Execute a callback with all reduce rules for a given encoding ID.
    pub(crate) fn apply_reduce_rules(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let exact = self.reduce_rules.get(&MatchKey::Array(array.encoding_id()));
        let any = self.reduce_rules.get(&MatchKey::Any);

        let rules = exact
            .iter()
            .chain(any.iter())
            .flat_map(|v| v.iter())
            .map(|v| v.as_ref());

        for rule in rules {
            if let Some(new_array) = rule.reduce(array)? {
                vortex_ensure!(
                    new_array.len() == array.len(),
                    "Parent reduction rule produced array of incorrect length: expected {}, got {}",
                    array.len(),
                    new_array.len()
                );
                #[cfg(debug_assertions)]
                vortex_ensure!(
                    new_array.dtype() == array.dtype(),
                    "Parent reduction rule produced array of incorrect dtype: expected {}, got {}",
                    array.dtype(),
                    new_array.dtype()
                );
                return Ok(Some(new_array));
            }
        }

        Ok(None)
    }

    /// Execute a callback with all parent reduce rules for a given child and parent encoding ID.
    ///
    /// Returns rules from both specific parent rules (if parent_id provided) and "any parent" wildcard rules.
    pub(crate) fn apply_parent_rules(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let exact_parent = self.parent_rules.get(&(
            MatchKey::Array(child.encoding_id()),
            MatchKey::Array(parent.encoding_id()),
        ));
        let any_parent = self
            .parent_rules
            .get(&(MatchKey::Array(child.encoding_id()), MatchKey::Any));
        let any_child = self
            .parent_rules
            .get(&(MatchKey::Any, MatchKey::Array(parent.encoding_id())));
        let any_both = self.parent_rules.get(&(MatchKey::Any, MatchKey::Any));

        let rules = exact_parent
            .iter()
            .chain(any_parent.iter())
            .chain(any_child.iter())
            .chain(any_both.iter())
            .flat_map(|v| v.iter())
            .map(|arc| arc.as_ref());

        for rule in rules {
            if let Some(new_array) = rule.reduce_parent(child, parent, child_idx)? {
                vortex_ensure!(
                    new_array.len() == parent.len(),
                    "Parent reduction rule produced array of incorrect length: expected {}, got {}",
                    parent.len(),
                    new_array.len()
                );
                #[cfg(debug_assertions)]
                vortex_ensure!(
                    new_array.dtype() == parent.dtype(),
                    "Parent reduction rule produced array of incorrect dtype: expected {}, got {}",
                    parent.dtype(),
                    new_array.dtype()
                );
                return Ok(Some(new_array));
            }
        }

        Ok(None)
    }
}
