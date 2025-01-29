use std::ops::Deref;
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
pub trait Accessor<T>: Send + Sync + Deref<Target = ArrayData> {
    fn batch_size(&self, start_idx: usize) -> usize {
        usize::min(ITER_BATCH_SIZE, self.len() - start_idx)
    }

    fn value_unchecked(&self, index: usize) -> T;

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
