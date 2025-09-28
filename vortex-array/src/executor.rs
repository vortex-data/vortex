// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, TryFutureExt};
use vortex_error::{
    SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err,
};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::arrays::ConstantArray;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, OperatorEq, OperatorHash, OperatorRef,
};
use crate::pipeline::operator::PipelineOperator;
use crate::{Canonical, IntoArray, ToCanonical};

/// An executor that runs an operator tree.
///
/// The executor performs common subtree elimination by creating BatchExecution nodes that hold
/// shared futures to the underlying execution.
///
/// It also finds sub-graphs of operator operators and executes them as a operator.
#[derive(Default)]
pub struct Executor {
    /// Cache of shared futures for common subtree elimination.
    /// We use WeakShared to allow futures to be dropped when no longer needed.
    execution_cache:
        HashMap<ProjectionKey, WeakShared<BoxFuture<'static, SharedVortexResult<Canonical>>>>,
}

impl Executor {
    /// Returns a projection future for the given operator.
    pub fn project(
        &mut self,
        operator: &OperatorRef,
        mask: Option<&OperatorRef>,
    ) -> BoxFuture<'static, VortexResult<Canonical>> {
        let execution = self.batch_projection(operator, mask);
        async move { Box::new(execution?).execute().await }.boxed()
    }

    pub fn project_mask(
        &mut self,
        operator_ref: &OperatorRef,
        mask: Option<&Mask>,
    ) -> BoxFuture<'static, VortexResult<Canonical>> {
        let mask: Option<OperatorRef> =
            mask.map(|mask| Arc::new(mask.clone().into_array().to_bool()) as OperatorRef);
        self.project(operator_ref, mask.as_ref())
    }

    fn batch_projection(
        &mut self,
        operator: &OperatorRef,
        mask: Option<&OperatorRef>,
    ) -> VortexResult<SharedBatchExecution> {
        // FIXME(ngates): we should have a separate optimize call that turns the operator tree
        //  into a DAG by inserting shared CSE nodes, each of which has the ability to construct
        //  a shared execution future... somehow...

        // Check if we already have a shared future for this operator
        let key = ProjectionKey {
            operator: operator.clone(),
            mask: mask.cloned(),
        };
        if let Some(weak_shared) = self.execution_cache.get(&key) {
            if let Some(shared) = weak_shared.upgrade() {
                // Return a SharedBatchExecution that references the existing shared future
                return Ok(SharedBatchExecution(shared));
            } else {
                // If the weak reference is dead, remove it from the cache
                self.execution_cache.remove(&key);
            }
        }

        // Attempt to convert the operator into a pipeline operator, if so we use that to execute.
        //
        // The construction of this operator pulls the largest subgraph of nodes that can be
        // executed in a pipelined fashion.
        let operator = match PipelineOperator::new(operator.clone()) {
            None => operator.clone(),
            Some(pipeline_op) => Arc::new(pipeline_op),
        };

        log::info!("Executing operator: {}", operator.display_tree());

        let all_true_mask: OperatorRef = Arc::new(ConstantArray::new(true, operator.len()));

        let execution = operator
            .as_batch()
            .ok_or_else(|| {
                vortex_err!(
                    "Operator does not support batch execution OR pipelined execution: {:?}",
                    operator
                )
            })?
            .project(mask.unwrap_or(&all_true_mask), self)?;

        Ok(self.shared(key, execution))
    }

    fn shared(&mut self, key: ProjectionKey, execution: BatchExecutionRef) -> SharedBatchExecution {
        let shared_future = execution.execute().map_err(Arc::new).boxed().shared();
        self.execution_cache
            .insert(key, shared_future.downgrade().vortex_expect("just created"));
        SharedBatchExecution(shared_future)
    }
}

struct ProjectionKey {
    operator: OperatorRef,
    mask: Option<OperatorRef>,
}
impl Hash for ProjectionKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operator.operator_hash(state);
        if let Some(mask) = &self.mask {
            mask.operator_hash(state);
        };
    }
}
impl PartialEq for ProjectionKey {
    fn eq(&self, other: &Self) -> bool {
        if !self.operator.operator_eq(&other.operator) {
            return false;
        }
        match (&self.mask, &other.mask) {
            (Some(m1), Some(m2)) => m1.operator_eq(m2),
            (None, None) => true,
            _ => false,
        }
    }
}
impl Eq for ProjectionKey {}

impl BatchBindCtx for Executor {
    fn bind_project(
        &mut self,
        operator: &OperatorRef,
        mask: Option<&OperatorRef>,
    ) -> VortexResult<BatchExecutionRef> {
        if let Some(mask) = mask
            && mask.len() != operator.len()
        {
            vortex_bail!(
                "Mask length {} != operator length {}",
                mask.len(),
                operator.len()
            );
        }
        Ok(Box::new(self.batch_projection(operator, mask)?))
    }
}

/// A wrapper around a batch execution that makes it available for sharing across nodes within
/// common subtree elimination.
///
// TODO(ngates): I think we could turn this into a full operator so that the tree display
//  makes more sense? Currently we just perform CSE during execution, rather than an up-front
//  optimization.
#[derive(Clone)]
struct SharedBatchExecution(Shared<BoxFuture<'static, SharedVortexResult<Canonical>>>);

#[async_trait]
impl BatchExecution for SharedBatchExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        self.0.await.map_err(VortexError::from)
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use vortex_buffer::buffer;
    use vortex_metrics::VortexMetrics;

    use super::*;
    use crate::compute::Operator as Op;
    use crate::operator::compare::CompareOperator;
    use crate::operator::metrics::MetricsOperator;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_basic_execution() {
        let array = buffer![1i32, 2, 3, 4].into_array().to_primitive();

        let mut executor = Executor::default();
        let result =
            block_on(executor.project(&(Arc::new(array.clone()) as OperatorRef), None)).unwrap();
        assert_eq!(
            result.into_primitive().as_slice::<i32>(),
            array.as_slice::<i32>()
        );
    }

    #[test]
    fn test_pipelined_execution() {
        let lhs = buffer![1i32, 2, 3].into_array().to_primitive();
        let rhs = buffer![3i32, 2, 1].into_array().to_primitive();

        // The CompareOperator uses pipelined execution
        let compare: OperatorRef =
            Arc::new(CompareOperator::try_new(Arc::new(lhs), Arc::new(rhs), Op::Gt).unwrap());

        let mut executor = Executor::default();
        let result = block_on(executor.project(&compare, None)).unwrap();
        assert_eq!(
            result.into_bool().bool_vec().unwrap(),
            vec![false, false, true]
        );
    }

    #[test]
    fn test_common_subtree_elimination() {
        // We use the same array for lhs and rhs to check we eliminate the common subtree
        let array = buffer![1i32, 2, 3, 4].into_array().to_primitive();
        let array = Arc::new(MetricsOperator::new(
            Arc::new(array),
            VortexMetrics::default(),
        ));

        let compare =
            Arc::new(CompareOperator::try_new(array.clone(), array.clone(), Op::Gt).unwrap());
        let compare = Arc::new(MetricsOperator::new(compare, VortexMetrics::default()));

        let mut executor = Executor::default();
        let result = block_on(executor.project(&(compare.clone() as OperatorRef), None)).unwrap();
        assert_eq!(
            result.into_bool().bool_vec().unwrap(),
            vec![false, false, false, false]
        );

        // The comparison operator is pipelined, it also only gets executed once
        assert_eq!(compare.metrics().timer("operator.operator.step").count(), 1);
        // The array only gets executed once due to common subtree elimination
        assert_eq!(array.metrics().timer("operator.batch.execute").count(), 1);
    }
}
