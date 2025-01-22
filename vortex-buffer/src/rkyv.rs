use bytes::Bytes;
use rkyv::util::AlignedVec;

use crate::{Alignment, ByteBuffer};

impl<const A: usize> From<AlignedVec<A>> for ByteBuffer {
    fn from(value: AlignedVec<A>) -> Self {
        let alignment = Alignment::new(A);
        if value.is_empty() {
            return Self::empty_aligned(alignment);
        }
        Self::from_bytes_aligned(Bytes::from_owner(value), alignment)
    }
}
