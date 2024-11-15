use std::sync::Arc;

pub use adapter::*;
pub use ext::*;
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::validity::Validity;
use crate::ArrayData;

mod adapter;
mod ext;

pub const ITER_BATCH_SIZE: usize = 1024;

/// A stream of array chunks along with a DType.
/// Analogous to Arrow's RecordBatchReader.
pub trait ArrayIterator: Iterator<Item = VortexResult<ArrayData>> {
    fn dtype(&self) -> &DType;
}

pub type AccessorRef<T> = Arc<dyn Accessor<T>>;

/// Define the basic behavior required for batched iterators
pub trait Accessor<T>: Send + Sync {
    fn batch_size(&self, start_idx: usize) -> usize {
        usize::min(ITER_BATCH_SIZE, self.array_len() - start_idx)
    }
    fn array_len(&self) -> usize;
    fn is_valid(&self, index: usize) -> bool;
    fn value_unchecked(&self, index: usize) -> T;
    fn array_validity(&self) -> Validity;

    #[inline]
    fn decode_batch(&self, start_idx: usize) -> Vec<T> {
        let batch_size = self.batch_size(start_idx);

        let mut batch = Vec::with_capacity(batch_size);

        for (idx, batch_item) in batch
            .spare_capacity_mut()
            .iter_mut()
            .enumerate()
            .take(batch_size)
        {
            batch_item.write(self.value_unchecked(start_idx + idx));
        }

        // Safety:
        // We've made sure that we have at least `batch_size` elements to put into
        // the vector and sufficient capacity.
        unsafe {
            batch.set_len(batch_size);
        }

        batch
    }
}

/// Iterate over batches of compressed arrays, should help with writing vectorized code.
///
/// Note that it doesn't respect per-item validity, and the per-item `Validity` instance should be advised
/// for correctness, must "high-performance" code will ignore the validity when doing work, and will only
/// re-use it when reconstructing the result array.
pub struct VectorizedArrayIter<T> {
    accessor: AccessorRef<T>,
    validity: Validity,
    current_idx: usize,
    len: usize,
}

impl<T> VectorizedArrayIter<T>
where
    T: Copy,
{
    pub fn new(accessor: Arc<dyn Accessor<T>>) -> Self {
        let len = accessor.array_len();
        let validity = accessor.array_validity();

        Self {
            accessor,
            len,
            validity,
            current_idx: 0,
        }
    }
}

pub struct Batch<T> {
    data: Vec<T>,
    validity: Validity,
}

impl<T: Copy> Batch<T> {
    pub fn new(data: &[T], validity: Validity) -> Self {
        Self {
            data: data.to_vec(),
            validity,
        }
    }
}

impl<T> Batch<T> {
    pub fn new_from_vec(data: Vec<T>, validity: Validity) -> Self {
        Self { data, validity }
    }

    #[inline]
    pub fn is_valid(&self, index: usize) -> bool {
        self.validity.is_valid(index)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// `index` must be smaller than the batch's length.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.data.get_unchecked(index) }
    }

    pub fn as_<N: NativePType>(self) -> Batch<N> {
        assert_eq!(size_of::<T>(), size_of::<N>());
        Batch {
            data: unsafe { std::mem::transmute::<Vec<T>, Vec<N>>(self.data) },
            validity: self.validity,
        }
    }

    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }
}

pub struct FlattenedBatch<T> {
    inner: Batch<T>,
    current: usize,
}

impl<T> Iterator for FlattenedBatch<T>
where
    T: Copy,
{
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.inner.len() {
            None
        } else if !self.inner.is_valid(self.current) {
            self.current += 1;
            Some(None)
        } else {
            let old = self.current;
            self.current += 1;
            // Safety: We verified that this value is both valid and within the array's bounds
            Some(Some(unsafe { *self.inner.get_unchecked(old) }))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.inner.len() - self.current,
            Some(self.inner.len() - self.current),
        )
    }
}

impl<T: Copy> ExactSizeIterator for FlattenedBatch<T> {}

impl<T: Copy> IntoIterator for Batch<T> {
    type Item = Option<T>;
    type IntoIter = FlattenedBatch<T>;

    fn into_iter(self) -> Self::IntoIter {
        FlattenedBatch {
            inner: self,
            current: 0,
        }
    }
}

impl<T: Copy> Iterator for VectorizedArrayIter<T> {
    type Item = Batch<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx == self.accessor.array_len() {
            None
        } else {
            let data = self.accessor.decode_batch(self.current_idx);

            let validity = self
                .validity
                .slice(self.current_idx, self.current_idx + data.len())
                .vortex_expect("The slice bounds should always be within the array's limits");
            self.current_idx += data.len();

            let batch = Batch::new_from_vec(data, validity);

            Some(batch)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.len - self.current_idx,
            Some(self.len - self.current_idx),
        )
    }
}

impl<T: Copy> ExactSizeIterator for VectorizedArrayIter<T> {}
