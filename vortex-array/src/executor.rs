// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use vortex_error::{
    SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err,
};
use vortex_utils::aliases::hash_map::HashMap;

use crate::Canonical;
use crate::operator::{BatchBindCtx, BatchExecution, BatchExecutionRef, OperatorKey, OperatorRef};
use crate::pipeline::operator::PipelineOperator;

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
    execution_cache: HashMap<
        OperatorKey<OperatorRef>,
        WeakShared<BoxFuture<'static, SharedVortexResult<Canonical>>>,
    >,
}

impl Executor {
    /// Returns an execution future for the given operator.
    pub fn execute(
        &mut self,
        operator: OperatorRef,
    ) -> BoxFuture<'static, VortexResult<Canonical>> {
        let execution = self.batch_execution(&operator);
        async move { execution?.execute().await }.boxed()
    }

    fn batch_execution(&mut self, operator: &OperatorRef) -> VortexResult<BatchExecutionRef> {
        // FIXME(ngates): we should have a separate optimize call that turns the operator tree
        //  into a DAG by inserting shared CSE nodes, each of which has the ability to construct
        //  a shared execution future... somehow...

        // Check if we already have a shared future for this operator
        let key = OperatorKey(operator.clone());
        if let Some(weak_shared) = self.execution_cache.get(&key) {
            if let Some(shared) = weak_shared.upgrade() {
                // Return a SharedBatchExecution that references the existing shared future
                return Ok(Box::new(SharedBatchExecution(shared)));
            } else {
                // If the weak reference is dead, remove it from the cache
                self.execution_cache.remove(&key);
            }
        }

        // Attempt to convert the operator into a operator operator, if so we use that to execute.
        //
        // The construction of this operator pulls the largest subgraph of nodes that can be
        // executed in a pipelined fashion.
        let operator = match PipelineOperator::new(operator.clone()) {
            None => operator.clone(),
            Some(pipeline_op) => Arc::new(pipeline_op),
        };

        log::info!("Executing operator: {}", operator.display_tree());
        println!("Executing operator: {}", operator.display_tree());

        // For each child, create a batch execution that uses the executor to compute it.
        let mut children: Vec<_> = operator
            .children()
            .iter()
            .map(|child| self.batch_execution(child))
            .map_ok(Some)
            .try_collect()?;

        let execution = operator
            .as_batch()
            .ok_or_else(|| {
                vortex_err!(
                    "Operator does not support batch execution OR pipelined execution: {:?}",
                    operator
                )
            })?
            .bind(&mut children)?;

        let shared_future = execution.execute().map_err(Arc::new).boxed().shared();
        self.execution_cache.insert(
            OperatorKey(operator),
            shared_future.downgrade().vortex_expect("just created"),
        );
        Ok(Box::new(SharedBatchExecution(shared_future)))
    }
}

impl BatchBindCtx for Vec<Option<BatchExecutionRef>> {
    fn child(&mut self, idx: usize) -> VortexResult<BatchExecutionRef> {
        if idx >= self.len() {
            vortex_bail!("Child index {} out of bounds", idx);
        }
        self[idx]
            .take()
            .ok_or_else(|| vortex_err!("Child already consumed"))
    }
}

/// A wrapper around a batch execution that makes it available for sharing across nodes within
/// common subtree elimination.
///
// TODO(ngates): I think we could turn this into a full operator so that the tree display
//  makes more sense? Currently we just perform CSE during execution, rather than an up-front
//  optimization.
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
        let result = block_on(executor.execute(Arc::new(array.clone()))).unwrap();
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
        let compare =
            Arc::new(CompareOperator::try_new(Arc::new(lhs), Arc::new(rhs), Op::Gt).unwrap());

        let mut executor = Executor::default();
        let result = block_on(executor.execute(compare)).unwrap();
        assert_eq!(result.into_bool().bool_vec(), vec![false, false, true]);
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
        let result = block_on(executor.execute(compare.clone())).unwrap();
        assert_eq!(
            result.into_bool().bool_vec(),
            vec![false, false, false, false]
        );

        // The comparison operator is pipelined, it also only gets executed once
        assert_eq!(compare.metrics().timer("operator.operator.step").count(), 1);
        // The array only gets executed once due to common subtree elimination
        assert_eq!(array.metrics().timer("operator.batch.execute").count(), 1);
    }
}
