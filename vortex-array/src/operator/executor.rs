// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::operator::{BatchBindCtx, BatchExecution, BatchExecutionRef, OperatorRef};
use crate::pipeline::operator::PipelineOperator;
use crate::Canonical;

use async_trait::async_trait;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;

use vortex_error::{
    vortex_bail, vortex_err, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};
use vortex_utils::aliases::hash_map::HashMap;

/// An executor that runs an operator tree.
///
/// The executor performs common subtree elimination by creating BatchExecution nodes that hold
/// shared futures to the underlying execution.
///
/// It also finds sub-graphs of pipeline operators and executes them as a [`Pipeline`]
#[derive(Default)]
pub struct Executor {
    /// Cache of shared futures for common subtree elimination.
    /// We use WeakShared to allow futures to be dropped when no longer needed.
    execution_cache:
        HashMap<OperatorRef, WeakShared<BoxFuture<'static, SharedVortexResult<Canonical>>>>,
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
        // Check if we already have a shared future for this operator
        if let Some(weak_shared) = self.execution_cache.get(operator) {
            if let Some(shared) = weak_shared.upgrade() {
                // Return a SharedBatchExecution that references the existing shared future
                return Ok(Box::new(SharedBatchExecution(shared)));
            } else {
                // If the weak reference is dead, remove it from the cache
                self.execution_cache.remove(operator);
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
                vortex_err!("Operator does not support batch execution OR pipelined execution")
            })?
            .bind(&mut children)?;

        let shared_future = execution.execute().map_err(Arc::new).boxed().shared();
        self.execution_cache.insert(
            operator.clone(),
            shared_future.downgrade().vortex_expect("just created"),
        );
        Ok(Box::new(SharedBatchExecution(shared_future)))
    }
}

impl BatchBindCtx for Vec<Option<BatchExecutionRef>> {
    fn take_child(&mut self, idx: usize) -> VortexResult<BatchExecutionRef> {
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
struct SharedBatchExecution(Shared<BoxFuture<'static, SharedVortexResult<Canonical>>>);

#[async_trait]
impl BatchExecution for SharedBatchExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        self.0.await.map_err(VortexError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Operator;
    use crate::operator::compare::CompareOperator;
    use crate::{IntoArray, ToCanonical};
    use futures::executor::block_on;
    use vortex_buffer::buffer;

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
        let array = buffer![1i32, 2, 3, 4].into_array().to_primitive();

        // The CompareOperator uses pipelined execution
        let compare = CompareOperator::try_new(
            Arc::new(array.clone()),
            Arc::new(array.clone()),
            Operator::Gt,
        )
        .unwrap();

        let mut executor = Executor::default();
        let result = block_on(executor.execute(compare)).unwrap();
        assert_eq!(
            result.into_bool().bool_vec().unwrap(),
            vec![false, false, false, false]
        );
    }

    //
    // #[test]
    // fn test_common_subtree_elimination() {
    //     futures::executor::block_on(async {
    //         let exec_count = Arc::new(AtomicUsize::new(0));
    //         let operator = Arc::new(TestOperator {
    //             id: "shared_op".to_string(),
    //             value: 123,
    //             len: 5,
    //             execution_count: exec_count.clone(),
    //         }) as OperatorRef;
    //
    //         let mut executor = Executor::new();
    //
    //         // Execute the same operator twice
    //         let result1 = executor.execute(operator.clone()).await.unwrap();
    //         let result2 = executor.execute(operator.clone()).await.unwrap();
    //
    //         // Both results should be the same
    //         if let (Canonical::Primitive(array1), Canonical::Primitive(array2)) = (result1, result2)
    //         {
    //             assert_eq!(array1.len(), array2.len());
    //
    //             // Verify execution happened only once due to caching
    //             assert_eq!(
    //                 exec_count.load(Ordering::SeqCst),
    //                 1,
    //                 "Operator should only execute once due to caching"
    //             );
    //         } else {
    //             panic!("Expected Primitive canonical arrays");
    //         }
    //     });
    // }
}
