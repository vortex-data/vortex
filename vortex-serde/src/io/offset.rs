use std::future::Future;

use bytes::BytesMut;
use futures::FutureExt;

use crate::io::VortexReadAt;

/// An adapter that offsets all reads by a fixed amount.
pub struct OffsetReadAt<R> {
    read: R,
    offset: u64,
}

impl<R> Clone for OffsetReadAt<R>
where
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            read: self.read.clone(),
            offset: self.offset,
        }
    }
}

impl<R: VortexReadAt> OffsetReadAt<R> {
    pub fn new(read: R, offset: u64) -> Self {
        Self { read, offset }
    }
}

impl<R: VortexReadAt> VortexReadAt for OffsetReadAt<R> {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = std::io::Result<BytesMut>> + 'static {
        self.read.read_at_into(pos + self.offset, buffer)
    }

    fn performance_hint(&self) -> usize {
        self.read.performance_hint()
    }

    fn size(&self) -> impl Future<Output = u64> + 'static {
        let offset = self.offset;
        self.read.size().map(move |len| len - offset)
    }
}
