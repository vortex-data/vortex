use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::FutureExt;

use crate::io::VortexReadAt;

/// An adapter that offsets all reads by a fixed amount.
pub struct OffsetReadAt<R> {
    read: R,
    offset: u64,
}

impl<R: VortexReadAt> OffsetReadAt<R> {
    pub fn new(read: R, offset: u64) -> Self {
        Self { read, offset }
    }
}

impl<R: VortexReadAt> VortexReadAt for OffsetReadAt<R> {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> BoxFuture<std::io::Result<BytesMut>> {
        self.read.read_at_into(pos + self.offset, buffer)
    }

    fn performance_hint(&self) -> usize {
        self.read.performance_hint()
    }

    fn size(&self) -> BoxFuture<u64> {
        async move { self.read.size().await - self.offset }.boxed()
    }
}
