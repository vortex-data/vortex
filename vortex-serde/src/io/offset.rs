use std::future::Future;
use std::io;

use bytes::BytesMut;

use super::BufResult;
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
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> impl Future<Output = BufResult<()>> {
        self.read.read_at_into(pos + self.offset, buffer)
    }

    fn performance_hint(&self) -> usize {
        self.read.performance_hint()
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.read.size().await? - self.offset)
    }
}
