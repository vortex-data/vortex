use std::future::Future;

use futures_util::TryStreamExt;
use vortex_error::VortexResult;

use crate::array::ChunkedArray;
use crate::stream::take_rows::TakeRows;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::{Array, IntoArray};

pub trait ArrayStreamExt: ArrayStream {
    /// Collect the stream into a single `Array`.
    ///
    /// If the stream yields multiple chunks, they will be returned as a [`ChunkedArray`].
    fn into_array(self) -> impl Future<Output = VortexResult<Array>>
    where
        Self: Sized,
    {
        async move {
            let dtype = self.dtype().clone();
            let mut chunks: Vec<Array> = self.try_collect().await?;
            if chunks.len() == 1 {
                Ok(chunks.remove(0))
            } else {
                Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
            }
        }
    }

    /// Perform a row-wise selection on the stream from an array of sorted indicessss.
    fn take_rows(self, indices: Array) -> VortexResult<impl ArrayStream>
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
