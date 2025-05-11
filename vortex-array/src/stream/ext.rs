use std::future::Future;

use futures_util::TryStreamExt;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ChunkedArray;
use crate::stream::{ArrayStream, SendableArrayStream};

pub trait ArrayStreamExt: ArrayStream {
    /// Box the [`ArrayStream`] so that it can be sent between threads.
    fn boxed(self) -> SendableArrayStream
    where
        Self: Sized + Send + 'static,
    {
        Box::pin(self)
    }

    /// Collect the stream into a single `Array`.
    ///
    /// If the stream yields multiple chunks, they will be returned as a [`ChunkedArray`].
    fn read_all(self) -> impl Future<Output = VortexResult<ArrayRef>>
    where
        Self: Sized,
    {
        async move {
            let dtype = self.dtype().clone();
            let mut chunks: Vec<ArrayRef> = self.try_collect().await?;
            if chunks.len() == 1 {
                Ok(chunks.remove(0))
            } else {
                Ok(ChunkedArray::try_new(chunks, dtype)?.to_array())
            }
        }
    }
}

impl<S: ArrayStream> ArrayStreamExt for S {}
