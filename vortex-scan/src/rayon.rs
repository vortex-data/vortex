// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::ScanBuilder;
use crate::work_queue::{TaskFactory, WorkStealingQueue};
use crate::work_stealing_iter::{ArrayTask, WorkStealingArrayIterator};

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

impl ScanBuilder<ArrayRef> {
    /// Returns a [`ParallelArrayIterator`] driven by all threads of the installed Rayon thread
    /// pool.
    pub fn into_par_iter(self) -> VortexResult<impl ParallelArrayIterator + Send + 'static> {
        use ::rayon::prelude::*;

        let dtype = self.dtype()?;
        let arc_dtype = Arc::new(dtype.clone());
        let concurrency = self.concurrency;
        let tasks = self.build()?;
        let queue = WorkStealingQueue::new([Box::new(move || Ok(tasks)) as TaskFactory<ArrayTask>]);

        // We create one work-stealing iterator per rayon thread, which allows each thread to drive
        // work in parallel. The user can decide what to do with the results, e.g. mapping a
        // parallel iterator to convert from Vortex to Arrow will still run the conversion on
        // the thread pool.
        let par_iter = (0..rayon::current_num_threads())
            .into_par_iter()
            .flat_map_iter(move |_thread_id| {
                WorkStealingArrayIterator::new(queue.clone(), arc_dtype.clone(), concurrency)
            });

        Ok(ParallelArrayIteratorAdapter::new(dtype, par_iter))
    }
}
