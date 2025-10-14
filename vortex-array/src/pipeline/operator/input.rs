// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::operator::{LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef};
use crate::pipeline::bits::BitView;
use crate::pipeline::vec::Selection;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{
    BatchId, BindContext, Element, Kernel, KernelContext, N, PipelinedOperator, RowSelection,
};

/// An operator that exports a child operator's data in canonical pipelined form.
#[derive(Debug, Clone)]
pub struct PipelineInputOperator {
    child: OperatorRef,
}

impl OperatorHash for PipelineInputOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.child.operator_hash(state);
    }
}

impl OperatorEq for PipelineInputOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.child.operator_eq(&other.child)
    }
}

impl PipelineInputOperator {
    pub(super) fn new(child: OperatorRef) -> Self {
        Self { child }
    }
}

impl Operator for PipelineInputOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.operator.canonical")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn bounds(&self) -> LengthBounds {
        self.child.bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(PipelineInputOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
        }))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

impl PipelinedOperator for PipelineInputOperator {
    fn row_selection(&self) -> RowSelection {
        RowSelection::All
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let batch_id = ctx.batch_inputs()[0];
        if let DType::Primitive(ptype, _) = self.dtype() {
            match_each_native_ptype!(ptype, |T| {
                return Ok(Box::new(CanonicalPrimitiveKernel::<T> {
                    batch_id,
                    _phantom: PhantomData,
                }) as Box<dyn Kernel>);
            })
        }
        vortex_bail!("CanonicalPipelineOperator currently only supports primitive dtypes");
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![0]
    }
}

// FIXME(ngates): we should support canonical inputs to the operator to avoid copying.
struct CanonicalPrimitiveKernel<T> {
    batch_id: BatchId,
    _phantom: PhantomData<T>,
}

impl<T: Element + NativePType> Kernel for CanonicalPrimitiveKernel<T> {
    fn step(
        &self,
        ctx: &KernelContext,
        chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        // TODO(ngates): maybe we defer binding until execution time when we can pass in the
        //  canonical array directly? It would avoid this clone on each step.
        let array = ctx.batch_input(self.batch_id).as_primitive().buffer::<T>();

        // TODO(ngates): decide when to iterate set indices vs copy all values.

        let len = (array.len() - (chunk_idx * N)).min(N);
        // TODO(ngates): is this faster if we hard-code N?
        out.as_array_mut()[..len].copy_from_slice(&array[chunk_idx * N..][..len]);

        // We don't know whether all true bits are at the front of the mask, so we must set
        // the selection to Mask.
        match selection.true_count() {
            N | 0 => out.set_selection(Selection::Prefix),
            _ => out.set_selection(Selection::Mask),
        }

        Ok(())
    }
}
