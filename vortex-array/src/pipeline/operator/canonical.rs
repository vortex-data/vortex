// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchId, BindContext, Operator, OperatorId, OperatorRef, PipelinedOperator,
};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N};
use std::any::Any;
use std::sync::Arc;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

/// An operator that exports a child operator's data in canonical pipelined form.
#[derive(Debug, Clone, Hash)]
pub struct CanonicalPipelineOperator {
    child: OperatorRef,
}

impl PartialEq for CanonicalPipelineOperator {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
    }
}
impl Eq for CanonicalPipelineOperator {}

impl CanonicalPipelineOperator {
    pub(super) fn new(child: OperatorRef) -> Self {
        Self { child }
    }
}

impl Operator for CanonicalPipelineOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.pipeline.canonical")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn len(&self) -> usize {
        self.child.len()
    }

    fn children(&self) -> &[OperatorRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(CanonicalPipelineOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
        }))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

impl PipelinedOperator for CanonicalPipelineOperator {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let batch_id = ctx.batch_inputs()[0];
        if let DType::Primitive(ptype, _) = self.dtype() {
            match_each_native_ptype!(ptype, |T| {
                return Ok(Box::new(CanonicalPrimitiveKernel::<T> {
                    batch_id,
                    elements: None,
                    offset: 0,
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

// FIXME(ngates): we should support canonical inputs to the pipeline to avoid copying.
struct CanonicalPrimitiveKernel<T> {
    batch_id: BatchId,
    elements: Option<Buffer<T>>,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for CanonicalPrimitiveKernel<T> {
    fn step(&mut self, ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        if self.elements.is_none() {
            let array = ctx.batch_input(self.batch_id).clone().into_primitive();
            self.elements = Some(array.into_buffer());
        }

        let elements = self
            .elements
            .as_ref()
            .vortex_expect("elements not initialized");

        let len = (elements.len() - self.offset).min(N);
        out.set_len(len);
        out.as_slice_mut()
            .copy_from_slice(&elements[self.offset..][..len]);
        self.offset += len;

        Ok(())
    }
}
