// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_metrics::{Timer, VortexMetrics};

use crate::Canonical;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::pipeline::bits::BitView;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{BindContext, Kernel, KernelContext, PipelinedOperator, RowSelection};

/// An operator that wraps another operator and records metrics about its execution.
#[derive(Debug)]
pub struct MetricsOperator {
    inner: OperatorRef,
    metrics: VortexMetrics,
}

impl OperatorHash for MetricsOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.inner.operator_hash(state);
        // Include our ID just to differentiate from the inner operator
        self.id().hash(state);
    }
}

impl OperatorEq for MetricsOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.inner.operator_eq(&other.inner)
    }
}

impl MetricsOperator {
    pub fn new(inner: OperatorRef, metrics: VortexMetrics) -> Self {
        let metrics = metrics.child_with_tags([("operator", inner.id().as_ref().to_string())]);
        Self { inner, metrics }
    }

    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }
}

impl Operator for MetricsOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.metrics")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    fn bounds(&self) -> LengthBounds {
        self.inner.bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        self.inner.children()
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(MetricsOperator {
            inner: self.inner.clone().with_children(children)?,
            metrics: self.metrics.clone(),
        }))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        self.inner.as_batch().is_some().then_some(self)
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        // Only support pipelined execution if the inner operator does
        self.inner.as_pipelined().is_some().then_some(self)
    }
}

impl BatchOperator for MetricsOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        let inner = self.inner.as_batch().vortex_expect("checked").bind(ctx)?;
        let timer = self.metrics.timer("operator.batch.execute");
        Ok(Box::new(MetricsBatchExecution { inner, timer }))
    }
}

struct MetricsBatchExecution {
    inner: BatchExecutionRef,
    timer: Arc<Timer>,
}

#[async_trait]
impl BatchExecution for MetricsBatchExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let _timer = self.timer.time();
        self.inner.execute().await
    }
}

impl PipelinedOperator for MetricsOperator {
    fn row_selection(&self) -> RowSelection {
        self.inner
            .as_pipelined()
            .vortex_expect("checked")
            .row_selection()
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let inner = self
            .inner
            .as_pipelined()
            .vortex_expect("checked")
            .bind(ctx)?;
        let timer = self.metrics.timer("operator.operator.step");
        Ok(Box::new(MetricsKernel { inner, timer }))
    }

    fn vector_children(&self) -> Vec<usize> {
        self.inner
            .as_pipelined()
            .vortex_expect("checked")
            .vector_children()
    }

    fn batch_children(&self) -> Vec<usize> {
        self.inner
            .as_pipelined()
            .vortex_expect("checked")
            .batch_children()
    }
}

struct MetricsKernel {
    inner: Box<dyn Kernel>,
    timer: Arc<Timer>,
}

impl Kernel for MetricsKernel {
    fn step(
        &self,
        ctx: &KernelContext,
        chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let _timer = self.timer.time();
        self.inner.step(ctx, chunk_idx, selection, out)
    }
}
