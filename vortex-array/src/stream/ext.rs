use std::future::Future;

use futures_util::TryStreamExt;
use vortex_error::VortexResult;

use crate::array::ChunkedArray;
use crate::stream::take_rows::TakeRows;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::ArrayData;

pub trait ArrayStreamExt: ArrayStream {
    fn collect_chunked(self) -> impl Future<Output = VortexResult<ChunkedArray>>
    where
        Self: Sized,
    {
        async {
            let dtype = self.dtype().clone();
            self.try_collect()
                .await
                .and_then(|chunks| ChunkedArray::try_new(chunks, dtype))
        }
    }

    fn take_rows(self, indices: ArrayData) -> VortexResult<impl ArrayStream>
    where
        Self: Sized,
    {
        Ok(ArrayStreamAdapter::new(
            self.dtype().clone(),
            TakeRows::try_new(self, indices)?,
        ))
    }
}

impl<R: ArrayStream> ArrayStreamExt for R {}
