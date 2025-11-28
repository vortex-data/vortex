// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use vortex_error::VortexResult;

use crate::array::optimizer::rules::{
    AnyArrayParent, ArrayParentReduceRule, ArrayParentReduceRuleAdapter, ArrayReduceRule,
    ArrayReduceRuleAdapter, DynArrayParentReduceRule, DynArrayReduceRule,
};
use crate::array::ArrayRef;
use crate::vtable::{ArrayId, VTable};
use crate::ArrayVisitor;
use vortex_utils::aliases::hash_map::HashMap;

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
    reduce_rules: HashMap<ArrayId, Vec<Arc<dyn DynArrayReduceRule>>>,
    /// Parent reduce rules for specific parent types, indexed by (child_id, parent_id)
    parent_rules: HashMap<(ArrayId, ArrayId), Vec<Arc<dyn DynArrayParentReduceRule>>>,
    /// Wildcard parent rules (match any parent), indexed by child_id only
    any_parent_rules: HashMap<ArrayId, Vec<Arc<dyn DynArrayParentReduceRule>>>,
}

impl ArrayOptimizer {
    /// Optimize the given array by applying registered rewrite rules.
    ///
    /// This performs two passes following the ExprSession pattern:
    /// 1. Apply parent rules - bottom-up traversal checking parent-child relationships
    /// 2. Apply reduce rules - bottom-up traversal applying transformations to each node
    pub fn optimize_array(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        // First pass: apply parent rules
        let array = self.apply_parent_rules(array)?;

        // Second pass: apply reduce rules
        let array = self.apply_reduce_rules(array)?;

        Ok(array)
    }

    /// Apply parent rules in a bottom-up traversal.
    ///
    /// For each array, recursively process children first, then check if any parent
    /// rules apply to transform children based on their parent context.
    fn apply_parent_rules(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        // First, recursively apply parent rules to all children
        let children = array.children();
        if children.is_empty() {
            return Ok(array);
        }

        let mut optimized_children = Vec::with_capacity(children.len());
        let mut children_changed = false;

        for child in children.iter() {
            let optimized_child = self.apply_parent_rules(child.clone())?;
            children_changed |= !Arc::ptr_eq(&optimized_child, child);
            optimized_children.push(optimized_child);
        }

        // Reconstruct array with optimized children if any changed
        let array = if children_changed {
            array.with_children(&optimized_children)?
        } else {
            array
        };

        // Now try to apply parent rules to each optimized child in the context of this array
        // Use the optimized_children list directly instead of re-fetching from array.children()
        // let mut transformed_children = Vec::with_capacity(optimized_children.len());

        for (idx, child) in optimized_children.iter().enumerate() {
            let child_id = child.encoding_id();
            let parent_id = array.encoding_id();

            let result = self.with_parent_rules(
                &child_id,
                Some(&parent_id),
                |rules| -> VortexResult<Option<ArrayRef>> {
                    for rule in rules {
                        if let Some(new_array) = rule.reduce_parent(child, &array, idx)? {
                            return Ok(Some(new_array));
                        }
                    }
                    Ok(None)
                },
            )?;

            if let Some(transformed) = result {
                return Ok(transformed);
            }
        }

        // Reconstruct array with transformed children if any rules matched
        Ok(array)
    }

    /// Apply reduce rules in a bottom-up traversal.
    ///
    /// For each array, recursively process children first, then try to apply
    /// reduce rules to transform the array itself.
    fn apply_reduce_rules(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        // First, recursively apply reduce rules to all children
        let children = array.children();
        if !children.is_empty() {
            let mut new_children = Vec::with_capacity(children.len());
            let mut changed = false;

            for child in children.iter() {
                let optimized_child = self.apply_reduce_rules(child.clone())?;
                changed |= !Arc::ptr_eq(&optimized_child, child);
                new_children.push(optimized_child);
            }

            // Reconstruct array with optimized children if any changed
            let array = if changed {
                array.with_children(&new_children)?
            } else {
                array
            };

            // Now try to apply reduce rules to this array
            self.try_reduce(array)
        } else {
            // Leaf node - just try to reduce
            self.try_reduce(array)
        }
    }

    /// Try to apply reduce rules to a single array, recursively if a rule matches.
    fn try_reduce(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        let encoding_id = array.encoding_id();
        let result =
            self.with_reduce_rules(&encoding_id, |rules| -> VortexResult<Option<ArrayRef>> {
                for rule in rules {
                    if let Some(new_array) = rule.reduce(&array)? {
                        return Ok(Some(new_array));
                    }
                }
                Ok(None)
            })?;

        if let Some(transformed) = result {
            // Rule matched - recursively try to reduce the result
            // self.try_reduce(transformed)
            Ok(transformed)
        } else {
            Ok(array)
        }
    }

    /// Register a reduce rule for a specific array encoding.
    pub fn register_reduce_rule<V, R>(&mut self, vtable: &V, rule: R)
    where
        V: VTable,
        R: ArrayReduceRule<V> + 'static,
    {
        let adapter = ArrayReduceRuleAdapter::new(rule);
        let encoding_id = V::id(vtable);
        self.reduce_rules
            .entry(encoding_id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule for a specific parent type.
    pub fn register_parent_rule<Child, Parent, R>(
        &mut self,
        child_encoding: &Child,
        parent_encoding: &Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: ArrayParentReduceRule<Child, Parent> + 'static,
    {
        let adapter = ArrayParentReduceRuleAdapter::new(rule);
        let child_id = Child::id(child_encoding);
        let parent_id = Parent::id(parent_encoding);
        self.parent_rules
            .entry((child_id, parent_id))
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_any_parent_rule<Child, R>(&mut self, child_encoding: &Child, rule: R)
    where
        Child: VTable,
        R: ArrayParentReduceRule<Child, AnyArrayParent> + 'static,
    {
        let adapter = ArrayParentReduceRuleAdapter::new(rule);
        let child_id = Child::id(child_encoding);
        self.any_parent_rules
            .entry(child_id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Execute a callback with all reduce rules for a given encoding ID.
    pub(crate) fn with_reduce_rules<F, R>(&self, id: &ArrayId, f: F) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynArrayReduceRule>) -> R,
    {
        f(&mut self
            .reduce_rules
            .get(id)
            .iter()
            .flat_map(|v| v.iter())
            .map(|arc| arc.as_ref()))
    }

    /// Execute a callback with all parent reduce rules for a given child and parent encoding ID.
    ///
    /// Returns rules from both specific parent rules (if parent_id provided) and "any parent" wildcard rules.
    pub(crate) fn with_parent_rules<F, R>(
        &self,
        child_id: &ArrayId,
        parent_id: Option<&ArrayId>,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynArrayParentReduceRule>) -> R,
    {
        let specific_entry =
            parent_id.and_then(|pid| self.parent_rules.get(&(child_id.clone(), pid.clone())));
        let wildcard_entry = self.any_parent_rules.get(child_id);

        f(&mut specific_entry
            .iter()
            .flat_map(|v| v.iter())
            .chain(wildcard_entry.iter().flat_map(|v| v.iter()))
            .map(|arc| arc.as_ref()))
    }
}
