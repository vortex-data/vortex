// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::file::ReadRequest;

/// An I/O request, either a single read or a coalesced set of reads.
pub struct IoRequest(IoRequestInner);

impl IoRequest {
    pub(crate) fn new_single(request: ReadRequest) -> Self {
        IoRequest(IoRequestInner::Single(request))
    }

    pub(crate) fn new_coalesced(request: CoalescedRequest) -> Self {
        IoRequest(IoRequestInner::Coalesced(request))
    }

    pub fn offset(&self) -> u64 {
        match &self.0 {
            IoRequestInner::Single(r) => r.offset,
            IoRequestInner::Coalesced(r) => r.range.start,
        }
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            IoRequestInner::Single(r) => r.length,
            IoRequestInner::Coalesced(r) => usize::try_from(r.range.end - r.range.start)
                .vortex_expect("range too big for usize"),
        }
    }

    pub fn alignment(&self) -> Alignment {
        match &self.0 {
            IoRequestInner::Single(r) => r.alignment,
            IoRequestInner::Coalesced(r) => r.alignment,
        }
    }

    pub fn is_canceled(&self) -> bool {
        match &self.0 {
            IoRequestInner::Single(req) => req.callback.is_closed(),
            IoRequestInner::Coalesced(req) => req.requests.iter().all(|r| r.callback.is_closed()),
        }
    }

    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        match self.0 {
            IoRequestInner::Single(req) => req.resolve(result),
            IoRequestInner::Coalesced(req) => req.resolve(result),
        }
    }
}

pub enum IoRequestInner {
    Single(ReadRequest),
    Coalesced(CoalescedRequest),
}

/// A set of I/O requests that have been coalesced into a single larger request.
pub(crate) struct CoalescedRequest {
    pub range: Range<u64>,
    pub alignment: Alignment, // The alignment of the first request in the coalesced range.
    pub requests: Vec<ReadRequest>, // TODO(ngates): we could have enum of Single/Many to avoid Vec.
}

impl Debug for CoalescedRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
                    let start = (req.offset - self.range.start) as usize;
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
