use bytes::Bytes;
use rkyv::util::AlignedVec;

use crate::{Alignment, ByteBuffer};

impl<const A: usize> From<AlignedVec<A>> for ByteBuffer {
    fn from(value: AlignedVec<A>) -> Self {
        println!(
            "From<AlignedVec<A>> for ByteBuffer {} {}",
            A,
            value.as_ptr().align_offset(A)
        );
        if value.as_ptr().align_offset(A) != 0 {
            print!("Hmmm");
        }
        Self::from_bytes_aligned(Bytes::from_owner(value), Alignment::new(A))
    }
}
