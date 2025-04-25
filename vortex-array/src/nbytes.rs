use rust_lapper::{Interval, Lapper};

use crate::{Array, ArrayVisitorExt};

pub trait NBytes: Array {
    /// Total size of the array in bytes, including all children and buffers.
    // TODO(ngates): this should return u64
    fn nbytes(&self) -> usize {
        let mut intervals = Vec::new();
        for array in self.depth_first_traversal() {
            for buffer in array.buffers() {
                let slice: &[u8] = buffer.inner().as_ref();
                let start = slice.as_ptr() as usize;
                intervals.push(Interval {
                    start,
                    stop: start + slice.len(),
                    val: true,
                });
            }
        }
        Lapper::new(intervals).cov()
    }
}

impl<T: Array + ?Sized> NBytes for T {}
