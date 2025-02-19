use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::ChunkedArray;
use crate::iter::ArrayIterator;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::{Array, IntoArray};

pub trait ArrayIteratorExt: ArrayIterator {
    fn into_stream(self) -> impl ArrayStream
    where
        Self: Sized,
    {
        ArrayStreamAdapter::new(self.dtype().clone(), futures_util::stream::iter(self))
    }

    /// Collect the iterator into a single `Array`.
    ///
    /// If the iterator yields multiple chunks, they will be returned as a [`ChunkedArray`].
    fn into_array_data(self) -> VortexResult<Array>
    where
        Self: Sized,
    {
        let dtype = self.dtype().clone();
        let mut chunks: Vec<Array> = self.try_collect()?;
        if chunks.len() == 1 {
            Ok(chunks.remove(0))
        } else {
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
    }
}

impl<I: ArrayIterator> ArrayIteratorExt for I {}
