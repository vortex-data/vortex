// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult};

/// An I/O request, either a single read or a coalesced set of reads.
pub struct IoRequest(IoRequestInner);

impl IoRequest {
    pub(crate) fn new_single(request: ReadRequest) -> Self {
        IoRequest(IoRequestInner::Single(request))
    }

    pub(crate) fn new_coalesced(request: CoalescedRequest) -> Self {
        IoRequest(IoRequestInner::Coalesced(request))
    }

    // For debugging purposes.
    #[cfg(test)]
    pub(crate) fn inner(&self) -> &IoRequestInner {
        &self.0
    }

    /// Returns the starting offset of this request within the file.
    pub fn offset(&self) -> u64 {
        match &self.0 {
            IoRequestInner::Single(r) => r.offset,
            IoRequestInner::Coalesced(r) => r.range.start,
        }
    }

    /// Returns the byte range this request within the file.
    pub fn range(&self) -> Range<u64> {
        match &self.0 {
            IoRequestInner::Single(r) => {
                r.offset
                    ..(r.offset + u64::try_from(r.length).vortex_expect("length too big for u64"))
            }
            IoRequestInner::Coalesced(r) => r.range.clone(),
        }
    }

    /// Returns true if this request has zero length.
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            IoRequestInner::Single(r) => r.length == 0,
            IoRequestInner::Coalesced(r) => r.range.start == r.range.end,
        }
    }

    /// Returns the length of this request in bytes.
    pub fn len(&self) -> usize {
        match &self.0 {
            IoRequestInner::Single(r) => r.length,
            IoRequestInner::Coalesced(r) => usize::try_from(r.range.end - r.range.start)
                .vortex_expect("range too big for usize"),
        }
    }

    /// Returns the alignment requirement for this request.
    pub fn alignment(&self) -> Alignment {
        match &self.0 {
            IoRequestInner::Single(r) => r.alignment,
            IoRequestInner::Coalesced(r) => r.alignment,
        }
    }

    /// Returns true if all callbacks associated with this request have been dropped.
    /// In other words, there is no one waiting for the result of this request.
    pub fn is_canceled(&self) -> bool {
        match &self.0 {
            IoRequestInner::Single(req) => req.callback.is_closed(),
            IoRequestInner::Coalesced(req) => req.requests.iter().all(|r| r.callback.is_closed()),
        }
    }

    /// Resolves the request with the given result.
    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        match self.0 {
            IoRequestInner::Single(req) => req.resolve(result),
            IoRequestInner::Coalesced(req) => req.resolve(result),
        }
    }
}

pub(crate) enum IoRequestInner {
    Single(ReadRequest),
    Coalesced(CoalescedRequest),
}

pub(crate) type RequestId = usize;

pub(crate) struct ReadRequest {
    pub(crate) id: RequestId,
    pub(crate) offset: u64,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl Debug for ReadRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadRequest")
            .field("id", &self.id)
            .field("offset", &self.offset)
            .field("length", &self.length)
            .field("alignment", &self.alignment)
            .field("is_closed", &self.callback.is_closed())
            .finish()
    }
}

impl ReadRequest {
    pub(crate) fn resolve(self, result: VortexResult<ByteBuffer>) {
        if let Err(e) = self.callback.send(result) {
            log::debug!("ReadRequest {} dropped before resolving: {e}", self.id);
        }
    }
}

/// A set of I/O requests that have been coalesced into a single larger request.
pub(crate) struct CoalescedRequest {
    pub(crate) range: Range<u64>,
    pub(crate) alignment: Alignment, // The alignment of the first request in the coalesced range.
    pub(crate) requests: Vec<ReadRequest>, // TODO(ngates): we could have enum of Single/Many to avoid Vec.
}

impl Debug for CoalescedRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoalescedRequest")
            .field("#", &self.requests.len())
            .field("length", &(self.range.end - self.range.start))
            .field("range", &self.range)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl CoalescedRequest {
    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        match result {
            Ok(buffer) => {
                let buffer = buffer.aligned(Alignment::none());
                for req in self.requests.into_iter() {
                    let start = usize::try_from(req.offset - self.range.start)
                        .vortex_expect("invalid offset");
                    let end = start + req.length;
                    let slice = buffer.slice(start..end).aligned(req.alignment);
                    req.resolve(Ok(slice));
                }
            }
            Err(e) => {
                let e = Arc::new(e);
                for req in self.requests.into_iter() {
                    req.resolve(Err(VortexError::from(e.clone())));
                }
            }
        }
    }
}
