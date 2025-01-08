//! Layout operations are analogous to array compute functions, but since layouts are lazy, we
//! need to wrap up the operation in a polling model.

pub mod cached;
pub mod map;
mod resolved;

use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentReader};

/// The response to polling an operation.
pub enum Poll<R> {
    /// The result of the operation.
    Some(R),
    /// The operation requires additional segments before it can make progress.
    NeedMore(Vec<SegmentId>),
}

/// Macro to simplify the common pattern of polling an operation.
/// Similar to the [`std::task::ready`] macro.
#[macro_export]
macro_rules! ready {
    ($e:expr) => {
        match $e? {
            $crate::operations::Poll::Some(t) => t,
            $crate::operations::Poll::NeedMore(segments) => {
                return Ok($crate::operations::Poll::NeedMore(segments));
            }
        }
    };
}

/// A trait for performing operations over a layout.
pub trait Operation {
    type Output;

    /// Attempts to return the result of this operation. If the operation cannot make progress, it
    /// returns a vec of additional data segments using [`Poll::NeedMore`].
    ///
    /// Note that after successfully returning `Poll::Some` the operation may fail on subsequent
    /// calls to `poll`.
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>>;
}

impl<R> Operation for Box<dyn Operation<Output = R>> {
    type Output = R;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        self.as_mut().poll(segments)
    }
}

pub trait OperationExt: Operation {
    /// Box the operation.
    fn boxed(self) -> Box<dyn Operation<Output = Self::Output> + 'static>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Cache the result of the operation so it can be polled multiple times.
    fn cached(self) -> cached::CachedOperation<Self>
    where
        Self: Sized,
        Self::Output: Clone,
    {
        cached::CachedOperation {
            op: self,
            value: None,
        }
    }

    /// Map the output of the operation.
    fn map<F, R>(self, f: F) -> map::MapOperation<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Output) -> R,
    {
        map::MapOperation {
            op: self,
            func: Some(f),
        }
    }
}

impl<T: Operation> OperationExt for T {}

/// Create an operation whose result is already resolved.
pub fn resolved<R>(result: R) -> resolved::ResolvedOperation<R> {
    resolved::ResolvedOperation {
        result: Some(result),
    }
}
