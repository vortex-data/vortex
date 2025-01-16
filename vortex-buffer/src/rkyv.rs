use bytes::Bytes;
use rkyv::util::AlignedVec;

use crate::{Alignment, ByteBuffer};

impl<const A: usize> From<AlignedVec<A>> for ByteBuffer {
    fn from(value: AlignedVec<A>) -> Self {
        Self::from_bytes_aligned(Bytes::from_owner(value), Alignment::new(A))
    }
}
