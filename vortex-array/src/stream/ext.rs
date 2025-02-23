use std::future::Future;

use futures_util::TryStreamExt;
use vortex_error::VortexResult;

use crate::arrays::ChunkedArray;
use crate::stream::take_rows::TakeRows;
use crate::stream::{ArrayStream, ArrayStreamAdapter, SendableArrayStream};
use crate::{Array, ArrayRef, IntoArray};

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
    fn into_array(self) -> impl Future<Output = VortexResult<ArrayRef>>
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

    /// Perform a row-wise selection on the stream from an array of sorted indicessss.
    fn take_rows(self, indices: ArrayRef) -> VortexResult<impl ArrayStream>
    where
        Self: Sized,
    {
        Ok(ArrayStreamAdapter::new(
            self.dtype().clone(),
            TakeRows::try_new(self, indices)?,
        ))
    }
}

impl<S: ArrayStream> ArrayStreamExt for S {}
