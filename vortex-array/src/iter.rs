//! Iterator over slices of an array, and related utilities.

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::ChunkedArray;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::{Array, IntoArray};

/// Iterator of array with a known [`DType`].
///
/// Its up to implementations to guarantee all arrays have the same [`DType`].
pub trait ArrayIterator: Iterator<Item = VortexResult<Array>> {
    fn dtype(&self) -> &DType;
}

pub struct ArrayIteratorAdapter<I> {
    dtype: DType,
    inner: I,
}

impl<I> ArrayIteratorAdapter<I> {
    pub fn new(dtype: DType, inner: I) -> Self {
        Self { dtype, inner }
    }
}

impl<I> Iterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<Array>>,
{
    type Item = VortexResult<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I> ArrayIterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<Array>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

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
