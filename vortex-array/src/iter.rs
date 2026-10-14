// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Iterator over slices of an array, and related utilities.

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::dtype::DType;
use crate::stream::ArrayStream;
use crate::stream::ArrayStreamAdapter;

/// Iterator of array with a known [`DType`].
///
/// It's up to implementations to guarantee all arrays have the same [`DType`].
pub trait ArrayIterator: Iterator<Item = VortexResult<ArrayRef>> {
    fn dtype(&self) -> &DType;
}

impl ArrayIterator for Box<dyn ArrayIterator + Send> {
    #[inline]
    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }
}

pub struct ArrayIteratorAdapter<I> {
    dtype: DType,
    inner: I,
}

impl<I> ArrayIteratorAdapter<I> {
    #[inline]
    pub fn new(dtype: DType, inner: I) -> Self {
        Self { dtype, inner }
    }
}

impl<I> Iterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<ArrayRef>>,
{
    type Item = VortexResult<ArrayRef>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I> ArrayIterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<ArrayRef>>,
{
    #[inline]
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub trait ArrayIteratorExt: ArrayIterator {
    fn into_array_stream(self) -> impl ArrayStream
    where
        Self: Sized,
    {
        ArrayStreamAdapter::new(self.dtype().clone(), futures::stream::iter(self))
    }

    /// Collect the iterator into a single `Array`.
    ///
    /// If the iterator yields multiple chunks, they will be returned as a [`ChunkedArray`].
    fn read_all(self) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        let dtype = self.dtype().clone();
        let mut chunks = self.peekable();
        let Some(first) = chunks.next().transpose()? else {
            return Ok(ChunkedArray::try_new([], dtype)?.into_array());
        };
        if chunks.peek().is_none() {
            return Ok(first);
        }
        let chunks = std::iter::once(Ok(first))
            .chain(chunks)
            .map(|chunk| -> VortexResult<_> {
                let chunk = chunk?;
                vortex_ensure!(chunk.dtype() == &dtype, MismatchedTypes: &dtype, chunk.dtype());
                Ok(chunk)
            });
        let expected_nchunks = chunks.size_hint().0;
        chunks.process_results(|chunks| {
            // SAFETY: the iterator validates the dtype of every successfully yielded chunk.
            unsafe { ChunkedArray::new_unchecked_sized(chunks, dtype.clone(), expected_nchunks) }
                .into_array()
        })
    }
}

impl<I: ArrayIterator> ArrayIteratorExt for I {}

impl ArrayRef {
    /// Create an [`ArrayIterator`] over the array.
    pub fn to_array_iterator(&self) -> impl ArrayIterator + 'static {
        let dtype = self.dtype().clone();
        let iter = if let Some(chunked) = self.as_opt::<Chunked>() {
            ArrayChunkIterator::Chunked(Arc::new(chunked.into_owned()), 0)
        } else {
            ArrayChunkIterator::Single(Some(self.clone()))
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ArrayChunkIterator::Single(array) => array.take().map(Ok),
            ArrayChunkIterator::Chunked(chunked, idx) => (*idx < chunked.nchunks()).then(|| {
                let chunk = chunked.chunk(*idx);
                *idx += 1;
                Ok(chunk.clone())
            }),
        }
    }
}
