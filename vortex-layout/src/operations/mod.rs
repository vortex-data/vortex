//! Layout operations are analogous to array compute functions, but since layouts are lazy, we
//! need to wrap up the operation in a polling model.

pub mod scan;
pub mod stats;

use std::fmt::Debug;

use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentReader};

/// The response to polling an operation.
pub enum Poll<R> {
    /// The result of the operation.
    Some(R),
    /// The operation requires additional segments before it can make progress.
    NeedMore(Vec<SegmentId>),
}

pub trait Operator {
    type Result;
}

/// A trait for performing operations over a layout.
pub trait Operation<O: Operator>: 'static + Send + Sync + Debug {
    /// Attempts to return the result of this operation. If the operation cannot make progress, it
    /// returns a vec of additional data segments using [`Poll::NeedMore`].
    ///
    /// Note that after returning `Poll::Some` the [`Operation`] should continue to return the same
    /// result on subsequent calls to `poll`.
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<O::Result>>;
}

pub trait OperationExt<O: Operator>: Operation<O> {
    /// Box the operation.
    fn boxed(self) -> Box<dyn Operation<O> + 'static>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

impl<O: Operator, T: Operation<O>> OperationExt<O> for T {}
