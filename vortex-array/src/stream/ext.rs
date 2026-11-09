// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;

use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::stream::ArrayStream;
use crate::stream::ArrayStreamAdapter;
use crate::stream::SendableArrayStream;

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
                Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
            }
        }
    }

    /// Yield at most `rows` rows of the stream, slicing the last array that crosses the limit.
    ///
    /// The stream is not polled again once the limit is reached.
    fn limit(self, rows: u64) -> impl ArrayStream
    where
        Self: Sized + Unpin,
    {
        let dtype = self.dtype().clone();
        let arrays = stream::try_unfold((self, rows), |(mut arrays, remaining)| async move {
            if remaining == 0 {
                return Ok(None);
            }

            let Some(array) = arrays.next().await.transpose()? else {
                return Ok(None);
            };

            let kept = usize::try_from(remaining).map_or(array.len(), |left| left.min(array.len()));
            let array = if kept == array.len() {
                array
            } else {
                array.slice(0..kept)?
            };
            Ok(Some((array, (arrays, remaining - kept as u64))))
        });
        ArrayStreamAdapter::new(dtype, arrays)
    }
}

impl<S: ArrayStream> ArrayStreamExt for S {}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::ArrayStreamExt;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::ChunkedArray;

    #[rstest]
    #[case::none(0, vec![])]
    #[case::inside_first(2, vec![2])]
    #[case::on_boundary(3, vec![3])]
    #[case::across(5, vec![3, 2])]
    #[case::past_end(100, vec![3, 3])]
    fn limit_slices_the_last_array(
        #[case] rows: u64,
        #[case] lens: Vec<usize>,
    ) -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![0u32, 1, 2].into_array(),
                buffer![3u32, 4, 5].into_array(),
            ],
            buffer![0u32].into_array().dtype().clone(),
        )?
        .into_array();

        let arrays: Vec<ArrayRef> =
            block_on(chunked.to_array_stream().boxed().limit(rows).try_collect())?;
        assert_eq!(arrays.iter().map(|a| a.len()).collect::<Vec<_>>(), lens);
        Ok(())
    }
}
