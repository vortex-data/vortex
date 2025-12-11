// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::hash_map::HashMap;

use crate::Array;
use crate::ArrayVisitor;
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
    pub fn optimize_array(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        // To handle large and bushy plan trees, we implement iterative optimizer passes.
        // We need to know how to do one step of optimizer here.

        let mut job_id = 0;
        let mut make_job = |array: ArrayRef| {
            let job = OptimizerJob {
                child_tasks: vec![],
                unoptimized_children: array.children(),
                #[cfg(debug_assertions)]
                dtype: array.dtype().clone(),
                array,
                id: job_id,
            };

            job_id += 1;

            job
        };

        struct OptimizerJob {
            id: usize,
            array: ArrayRef,
            unoptimized_children: Vec<ArrayRef>,
            child_tasks: Vec<usize>,
            #[cfg(debug_assertions)]
            dtype: DType,
        }

        // mapping of results
        let mut results: HashMap<usize, ArrayRef> = HashMap::new();
        let mut optimize_stack = VecDeque::new();

        // Stage the first piece of work.
        let root_job = make_job(array.clone());
        let root_job_id = root_job.id;
        optimize_stack.push_front(root_job);

        tracing::debug!("Starting array optimization\n{}", array.display_tree());

        'outer: while !optimize_stack.is_empty() {
            // Pop off another job. This is an array which may have several children that need
            // to be optimized before it can itself be optimized.
            let mut job = optimize_stack.pop_front().unwrap();

            if let Some(child) = job.unoptimized_children.pop() {
                // Make a new task for the next child that needs to be completed.
                let child_task = make_job(child);
                job.child_tasks.push(child_task.id);

                optimize_stack.push_front(job);
                optimize_stack.push_front(child_task);
                continue 'outer;
            }

            // No unoptimized children, let's collect the results of optimizing.
            let task_ids = mem::take(&mut job.child_tasks);
            let optimized_children = task_ids
                .into_iter()
                .map(|id| {
                    results
                        .remove(&id)
                        .expect("optimizer attempted to finish task before its child completed")
                })
                .collect::<Vec<_>>();

            let array = job.array.with_children(optimized_children)?;
            #[cfg(debug_assertions)]
            {
                debug_assert_eq!(array.dtype(), &job.dtype);
            }

            if let Some(new_array) = self.apply_reduce_rules(&array)? {
                #[cfg(debug_assertions)]
                {
                    debug_assert_eq!(new_array.dtype(), &job.dtype);
                }

                // Update the job with the same job ID, but new children and new array
                job.unoptimized_children = new_array.children();
                job.child_tasks.clear();
                job.array = new_array;
                optimize_stack.push_front(job);
                continue 'outer;
            }

            // Otherwise, we push the result into the stack instead here.
            results.insert(job.id, array);
        }

        let optimized = results
            .remove(&root_job_id)
            .expect("job queue completion without resolving job 0");
        tracing::debug!("Optimized array\n{}", optimized.display_tree());

        Ok(optimized)
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
