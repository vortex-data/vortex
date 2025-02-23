use crate::{Array, ArrayVisitorExt};

pub trait NBytes: Array {
    /// Total size of the array in bytes, including all children and buffers.
    // TODO(ngates): this should return u64
    fn nbytes(&self) -> usize {
        let mut nbytes = 0;
        for array in self.depth_first_traversal() {
            for buffer in array.buffers() {
                nbytes += buffer.len();
            }
            nbytes += array.metadata().map_or(0, |b| b.len());
        }
        nbytes
    }
}

impl<T: Array + ?Sized> NBytes for T {}
