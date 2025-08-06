// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::buffers::BufferHandle;
use crate::pipeline::nodes::BinaryOperator;
use crate::pipeline::nodes::plan::BindContext;
use crate::pipeline::nodes::plan::source::{SourceNode, SourceNodeAdapter, SourceOperator};
use crate::pipeline::nodes::vector::VectorMut;
use crate::pipeline::types::Element;
use crate::pipeline::vector::Vector;
use crate::pipeline::view::{TypedView, TypedViewMut};
use crate::pipeline::{N, PipelineContext};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::task::{Poll, ready};
use vortex_dtype::NativePType;
use vortex_error::VortexResult;

#[derive(Debug)]
pub struct PrimitiveSource<T: NativePType> {
    len: usize,
    buffer: BufferHandle<T>,
}

impl<T: Element + NativePType> PrimitiveSource<T> {
    /// Creates a new `PrimitiveSource` with the given length and buffer.
    pub fn new(
        len: usize,
        buffer: BufferHandle<T>,
    ) -> SourceNodeAdapter<T, PrimitiveSourceOperator<T>, Self> {
        SourceNodeAdapter::new(PrimitiveSource { len, buffer })
    }
}

impl<T: NativePType> Hash for PrimitiveSource<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
        self.len.hash(state);
    }
}

impl<T: Element + NativePType> SourceNode<T, PrimitiveSourceOperator<T>> for PrimitiveSource<T> {
    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<PrimitiveSourceOperator<T>> {
        Ok(PrimitiveSourceOperator {
            _len: self.len,
            buffer: self.buffer.clone(),
            _offset: 0,
        })
    }
}

/// A source node that produces a primitive type vector.
pub struct PrimitiveSourceOperator<T: NativePType> {
    _len: usize,
    buffer: BufferHandle<T>,
    offset: usize,
}

impl<T: Element + NativePType> SourceOperator<T> for PrimitiveSourceOperator<T> {
    fn execute_all(
        &mut self,
        ctx: &dyn PipelineContext,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;
        todo!()
    }
}

/// A compare operator for primitive types that compares two vectors element-wise using a binary
/// operation.
pub struct ComparePrimitive<T, Op> {
    op: Op,
    _phantom: PhantomData<T>,
}

impl<T: Element + NativePType, Op> BinaryOperator<T, T, bool> for ComparePrimitive<T, Op>
where
    Op: Fn(&T, &T) -> bool,
{
    fn execute(
        &mut self,
        _ctx: &dyn PipelineContext,
        lhs: &TypedView<T>,
        rhs: &TypedView<T>,
        out: &mut TypedViewMut<bool>,
    ) -> Poll<VortexResult<()>> {
        for i in 0..N {
            out.as_mut()[i] = (self.op)(&lhs.as_ref()[i], &rhs.as_ref()[i]);
        }
        Poll::Ready(Ok(()))
    }
}
