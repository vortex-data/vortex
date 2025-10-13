// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, MaskExecution, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::Canonical;
use async_trait::async_trait;
use futures::try_join;
use itertools::Itertools;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};

/// An operator that produces a boolean array from offsets and lengths of true runs.
#[derive(Debug)]
pub struct BoolRunsOperator {
    /// The total length of the output array.
    len: usize,
    /// The child operators offsets and lengths.
    children: [OperatorRef; 2],
}

impl BoolRunsOperator {
    pub fn new(len: usize, offsets: OperatorRef, lengths: OperatorRef) -> Self {
        assert_eq!(
            offsets.len(),
            lengths.len(),
            "Offsets and lengths must have the same length"
        );
        Self {
            len,
            children: [offsets, lengths],
        }
    }

    #[inline(always)]
    pub fn offsets(&self) -> &OperatorRef {
        &self.children[0]
    }

    #[inline(always)]
    pub fn lengths(&self) -> &OperatorRef {
        &self.children[1]
    }
}

impl OperatorHash for BoolRunsOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.offsets().operator_hash(state);
        self.lengths().operator_hash(state);
    }
}

impl OperatorEq for BoolRunsOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.offsets().operator_eq(&other.offsets())
            && self.lengths().operator_eq(&other.lengths())
    }
}

impl Operator for BoolRunsOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.bool_runs")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &DType::Bool(Nullability::NonNullable)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let (offsets, lengths) = children
            .into_iter()
            .collect_tuple()
            .vortex_expect("Expected 2 children for BoolRunsOperator");
        Ok(Arc::new(BoolRunsOperator::new(self.len, offsets, lengths)))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for BoolRunsOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let mask_exec = ctx.bind_mask(mask)?;

        // We cannot push-down the mask through run-length encoding
        let offsets_exec = ctx.bind_project(self.offsets(), None)?;
        let lengths_exec = ctx.bind_project(self.lengths(), None)?;

        Ok(Box::new(BoolRunsProjection {
            mask: mask_exec,
            offsets: offsets_exec,
            lengths: lengths_exec,
        }))
    }
}

struct BoolRunsProjection {
    mask: MaskExecution,
    offsets: BatchExecutionRef,
    lengths: BatchExecutionRef,
}

#[async_trait]
impl BatchExecution for BoolRunsProjection {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let (mask, offsets, lengths) =
            try_join!(self.mask, self.offsets.execute(), self.lengths.execute())?;
        todo!()
    }
}
