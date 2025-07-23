// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::ScanBuilder;

/// A trait for a Rayon parallel Array iterator over arrays with a known [`DType`].
pub trait ParallelArrayIterator: ParallelIterator<Item = VortexResult<ArrayRef>> {
    /// Returns the data type of the elements in the iterator.
    fn dtype(&self) -> &DType;
}

/// An adapter for a Rayon parallel iterator that implements [`ParallelArrayIterator`].
pub struct ParallelArrayIteratorAdapter<I> {
    dtype: DType,
    inner: I,
}

impl<I> ParallelArrayIteratorAdapter<I> {
    /// Creates a new `ParallelArrayIteratorAdapter` with the given data type and inner iterator.
    pub fn new(dtype: DType, inner: I) -> Self {
        Self { dtype, inner }
    }
}

impl<I> ParallelArrayIterator for ParallelArrayIteratorAdapter<I>
where
    I: ParallelIterator<Item = VortexResult<ArrayRef>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<I> ParallelIterator for ParallelArrayIteratorAdapter<I>
where
    I: ParallelIterator<Item = VortexResult<ArrayRef>>,
{
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.inner.drive_unindexed(consumer)
    }
}

pub trait ParallelArrayIteratorExt: ParallelArrayIterator {
    /// Collects all elements of the iterator into a single [`ArrayRef`].
    fn read_all(self) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        let dtype = self.dtype().clone();
        let chunks: Vec<_> = self
            .collect::<Vec<VortexResult<ArrayRef>>>()
            .into_iter()
            .try_collect()?;

        if chunks.len() == 1 {
            return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
        }

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }
}

impl<I> ParallelArrayIteratorExt for I where I: ParallelArrayIterator {}

impl ScanBuilder<ArrayRef> {
    /// Returns a [`ParallelArrayIterator`] driven by all threads of the installed Rayon thread
    /// pool.
    ///
    /// Note that this iterator does not currently perform any per-thread task concurrency.
    pub fn into_par_iter(self) -> VortexResult<impl ParallelArrayIterator + Send + 'static> {
        use ::rayon::prelude::*;

        let dtype = self.dtype()?;
        let tasks = self.build()?;

        let par_iter = tasks
            .into_par_iter()
            .filter_map(|task| futures::executor::block_on(task).transpose());

        Ok(ParallelArrayIteratorAdapter::new(dtype, par_iter))
    }
}
