use std::future::Future;
use std::io;

use bytes::Bytes;
use futures::FutureExt;

use crate::VortexReadAt;

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
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        self.read.read_byte_range(pos + self.offset, len)
    }

    fn performance_hint(&self) -> usize {
        self.read.performance_hint()
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let offset = self.offset;
        self.read.size().map(move |len| len.map(|len| len - offset))
    }
}
