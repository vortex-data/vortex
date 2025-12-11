// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::hash_map::HashMap;

use crate::Array;
use crate::ArrayVisitor;
use crate::ArrayVisitorExt;
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
    /// Optimize the given array by applying registered rewrite rules.
    ///
    // TODO(ngates): this is slow, overly recursive, and will stack overflow if the rules end up
    //  forming a cycle.
    pub fn optimize_array(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        // Inner recursive function that tracks number of iterations to avoid infinite loops.
        fn inner(
            opt: &ArrayOptimizer,
            array: &ArrayRef,
            iterations: usize,
        ) -> VortexResult<ArrayRef> {
            if iterations == 0 {
                // Prevent infinite recursion by limiting the number of iterations.
                return Ok(array.clone());
            }

            // TODO(ngates): we should reduce first on the way down?
            let new_children: Vec<_> = array
                .children()
                .iter()
                .map(|child| inner(opt, child, iterations - 1))
                .try_collect()?;

            // If any children changed, reconstruct the array
            let array = array.with_children(new_children)?;

            // Apply reduction rules to the current array until no more rules apply.
            if let Some(new_array) = opt.apply_reduce_rules(&array)? {
                // Start over
                return inner(opt, &new_array, iterations - 1);
            }

            // Apply parent reduction rules to each child in the context of the current array.
            for (idx, child) in array.children().iter().enumerate() {
                if let Some(new_array) = opt.apply_parent_rules(child, &array, idx)? {
                    // If the parent was replaced, then we start over with the new parent
                    return inner(opt, &new_array, iterations - 1);
                }
            }

            Ok(array)
        }

        // The number of iterations we allow is the number of nodes in the array tree * 4.
        // No real reason to pick 4.
        let max_iterations = array.depth_first_traversal().count() * 4;

        inner(self, array, max_iterations)
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
