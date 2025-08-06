// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod common;
mod operator;
mod pipeline;
mod plan;
mod vector;

use crate::pipeline::PipelineContext;
use crate::pipeline::types::Element;
use crate::pipeline::view::{TypedView, TypedViewMut};
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;
use std::task::Poll;
use vortex_error::VortexResult;

/// An operator that applies a scalar function mapping from an input type `A` to an output type `X`.
pub trait UnaryOperator<A: Element, X: Element> {
    fn execute(
        &mut self,
        ctx: &dyn PipelineContext,
        input: &TypedView<A>,
        out: &mut TypedViewMut<X>,
    ) -> Poll<VortexResult<()>>;
}

/// An operator that applies a scalar function by mutating an input of type `A` in place.
pub trait UnaryInPlaceOperator<X: Element> {
    fn execute(
        &mut self,
        ctx: &dyn PipelineContext,
        in_out: &mut TypedViewMut<X>,
    ) -> Poll<VortexResult<()>>;
}

/// An operator that applies a binary function mapping from two input types `A` and `B` to an
/// output type `X`.
pub trait BinaryOperator<A: Element, B: Element, X: Element> {
    fn execute(
        &mut self,
        ctx: &dyn PipelineContext,
        lhs: &TypedView<A>,
        rhs: &TypedView<B>,
        out: &mut TypedViewMut<X>,
    ) -> Poll<VortexResult<()>>;
}

/// An operator that applies a ternary function from three input types `A`, `B`, and `C` to an
/// output type `X`.
pub trait TernaryOperator<A: Element, B: Element, C: Element, X: Element> {
    fn execute(
        &mut self,
        ctx: &dyn PipelineContext,
        in_a: &TypedView<A>,
        in_b: &TypedView<B>,
        in_c: &TypedView<C>,
        out: &mut TypedViewMut<X>,
    ) -> Poll<VortexResult<()>>;
}

pub struct UnaryNode<A: Element, X: Element, O: UnaryOperator<A, X>> {
    id: NodeId,
    operator: O,
    _phantom: PhantomData<(A, X)>,
}

pub struct UnaryInPlaceNode<X: Element, O: UnaryInPlaceOperator<X>> {
    id: NodeId,
    operator: O,
    _phantom: PhantomData<X>,
}

pub struct BinaryNode<A: Element, B: Element, X: Element, O: BinaryOperator<A, B, X>> {
    id: NodeId,
    operator: O,
    _phantom: PhantomData<(A, B, X)>,
}

pub struct TernaryNode<
    A: Element,
    B: Element,
    C: Element,
    X: Element,
    O: TernaryOperator<A, B, C, X>,
> {
    operator: O,
    _phantom: PhantomData<(A, B, C, X)>,
}

static NODE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl Default for NodeId {
    fn default() -> Self {
        NodeId(NODE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[cfg(test)]
mod test {
    use crate::pipeline::buffers::BufferHandle;
    use crate::pipeline::nodes::common::PrimitiveSource;
    use crate::pipeline::nodes::pipeline::Pipeline;
    use crate::pipeline::nodes::plan::PlanNode;
    use crate::pipeline::vector::Vector;
    use std::cell::RefCell;
    use std::task::Poll;
    use vortex_buffer::buffer;
    use vortex_error::vortex_panic;

    #[test]
    fn test_pipeline() {
        // First, let's construct a simple pipeline with a unary operator.
        let data = buffer![0..10000];
        let src = PrimitiveSource::new(data.len(), BufferHandle::new(data.into_byte_buffer()));

        let mut out = RefCell::new(Vector::new_with_vtype(src.output_type()));

        let mut pipeline = Pipeline::new(&src).unwrap();
        let mut more_work = true;
        while more_work {
            more_work = match pipeline.step(&(), &mut out) {
                Poll::Ready(more_work) => more_work.unwrap(),
                Poll::Pending => {
                    vortex_panic!("Pending for in-memory pipeline")
                }
            };
            println!("OUTPUT: {:?}", out.borrow());
        }

        assert!(false);
    }
}
