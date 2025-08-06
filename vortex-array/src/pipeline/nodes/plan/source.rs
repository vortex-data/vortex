// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A source node takes no input vectors and produces a single output vector.

use crate::pipeline::bits::BitView;
use crate::pipeline::nodes::plan::{BindContext, PlanNode};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vector::{Vector, VectorRefMut};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Operator, PipelineContext};
use std::cell::Ref;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::task::Poll;
use vortex_error::VortexResult;

#[derive(Debug)]
pub struct SourceNodeAdapter<E: Element, Op: SourceOperator<E>, S: SourceNode<E, Op>> {
    source: S,
    _phantom: PhantomData<(E, Op)>,
}

impl<E: Element, Op: SourceOperator<E>, S: SourceNode<E, Op>> SourceNodeAdapter<E, Op, S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            _phantom: PhantomData,
        }
    }
}

impl<E: Element, Op: SourceOperator<E>, S: SourceNode<E, Op>> Hash for SourceNodeAdapter<E, Op, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl<E: Element, Op: SourceOperator<E>, S: SourceNode<E, Op>> PlanNode
    for SourceNodeAdapter<E, Op, S>
{
    fn output_type(&self) -> VType {
        E::vtype()
    }

    fn children(&self) -> &[Box<dyn PlanNode>] {
        &[]
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Operator>> {
        Ok(Box::new(SourceOperatorAdapter::new(self.source.bind(ctx)?)))
    }
}

pub trait SourceNode<E, Op>: Debug + Hash {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Op>;
}

pub trait SourceOperator<X: Element>: 'static {
    /// Execute with a mask that is all true.
    fn step_all_true(
        &mut self,
        ctx: &dyn PipelineContext,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>>;
}

pub struct SourceOperatorAdapter<X, Op> {
    op: Op,
    _phantom: PhantomData<X>,
}

impl<X, Op> SourceOperatorAdapter<X, Op> {
    pub fn new(op: Op) -> Self {
        Self {
            op,
            _phantom: PhantomData,
        }
    }
}

impl<X, Op> Operator for SourceOperatorAdapter<X, Op>
where
    X: Element,
    Op: SourceOperator<X>,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        todo!()
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        mask: BitView,
        output: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        println!("MASK: {:?}", mask);
        self.op.step_all_true(ctx, output)
    }
}
