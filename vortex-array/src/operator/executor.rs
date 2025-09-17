// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::HashMap;
use std::sync::Arc;

use crate::operator::{BatchBindCtx, BatchExecutionRef, OperatorRef};
use crate::pipeline::operator::PipelineOperator;
use crate::Canonical;

use futures::future::{BoxFuture, WeakShared};
use futures::FutureExt;
use itertools::Itertools;

use vortex_error::{vortex_bail, vortex_err, SharedVortexResult, VortexResult};

/// An executor that runs an operator tree.
///
/// The executor performs common subtree elimination by creating BatchExecution nodes that hold
/// shared futures to the underlying execution.
///
/// It also finds sub-graphs of pipeline operators and executes them as a [`Pipeline`]
#[derive(Default)]
pub struct Executor {
    #[allow(dead_code)]
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
        // Attempt to convert the operator into a pipeline operator, if so we use that to execute.
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

        operator
            .as_batch()
            .ok_or_else(|| {
                vortex_err!("Operator does not support batch execution OR pipelined execution")
            })?
            .bind(&mut children)
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

#[cfg(test)]
mod tests {
    use super::*;
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
