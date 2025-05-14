//! Iterator over slices of an array, and related utilities.

use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::{Array, ArrayRef, IntoArray};

/// Iterator of array with a known [`DType`].
///
/// It's up to implementations to guarantee all arrays have the same [`DType`].
pub trait ArrayIterator: Iterator<Item = VortexResult<ArrayRef>> {
    fn dtype(&self) -> &DType;
}

impl ArrayIterator for Box<dyn ArrayIterator + Send> {
    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }
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
    I: Iterator<Item = VortexResult<ArrayRef>>,
{
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I> ArrayIterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<ArrayRef>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub trait ArrayIteratorExt: ArrayIterator {
    fn into_array_stream(self) -> impl ArrayStream
    where
        Self: Sized,
    {
        ArrayStreamAdapter::new(self.dtype().clone(), futures_util::stream::iter(self))
    }

    /// Collect the iterator into a single `Array`.
    ///
    /// If the iterator yields multiple chunks, they will be returned as a [`ChunkedArray`].
    fn read_all(self) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        let dtype = self.dtype().clone();
        let mut chunks: Vec<ArrayRef> = self.try_collect()?;
        if chunks.len() == 1 {
            Ok(chunks.remove(0))
        } else {
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
    }
}

impl<I: ArrayIterator> ArrayIteratorExt for I {}

impl dyn Array + '_ {
    /// Create an [`ArrayIterator`] over the array.
    pub fn to_array_iterator(&self) -> impl ArrayIterator + 'static {
        let dtype = self.dtype().clone();
        let iter = if let Some(chunked) = self.as_opt::<ChunkedVTable>() {
            ArrayChunkIterator::Chunked(Arc::new(chunked.clone()), 0)
        } else {
            ArrayChunkIterator::Single(Some(self.to_array()))
        };
        ArrayIteratorAdapter::new(dtype, iter)
    }
}

/// We define a single iterator that can handle both chunked and non-chunked arrays.
/// This avoids the need to create boxed static iterators for the two chunked and non-chunked cases.
enum ArrayChunkIterator {
    Single(Option<ArrayRef>),
    Chunked(Arc<ChunkedArray>, usize),
}

impl Iterator for ArrayChunkIterator {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ArrayChunkIterator::Single(array) => array.take().map(Ok),
            ArrayChunkIterator::Chunked(chunked, idx) => (*idx < chunked.nchunks()).then(|| {
                let chunk = chunked.chunk(*idx).vortex_expect("not out of bounds");
                *idx += 1;
                Ok(chunk.clone())
            }),
        }
    }
}
