// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// An I/O request, either a single read or a coalesced set of reads.
pub(crate) struct IoRequest(IoRequestInner);

impl IoRequest {
    pub(crate) fn new_single(request: ReadRequest) -> Self {
        IoRequest(IoRequestInner::Single(request))
    }

    pub(crate) fn new_coalesced(request: CoalescedRequest) -> Self {
        IoRequest(IoRequestInner::Coalesced(request))
    }

    /// Returns the starting offset of this request within the file.
    pub fn offset(&self) -> u64 {
        match &self.0 {
            IoRequestInner::Single(r) => r.offset,
            IoRequestInner::Coalesced(r) => r.range.start,
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

    /// Resolves the request with the given result.
    pub fn resolve(self, result: VortexResult<BufferHandle>) {
        match self.0 {
            IoRequestInner::Single(req) => req.resolve(result),
            IoRequestInner::Coalesced(req) => req.resolve(result),
        }
    }
}

// Testing functionality
#[cfg(test)]
impl IoRequest {
    pub(crate) fn inner(&self) -> &IoRequestInner {
        &self.0
    }

    /// Returns the byte range this request within the file.
    pub(crate) fn range(&self) -> Range<u64> {
        match &self.0 {
            IoRequestInner::Single(r) => {
                r.offset
                    ..(r.offset + u64::try_from(r.length).vortex_expect("length too big for u64"))
            }
            IoRequestInner::Coalesced(r) => r.range.clone(),
        }
    }
}

pub(crate) enum IoRequestInner {
    Single(ReadRequest),
    Coalesced(CoalescedRequest),
}

pub(crate) type RequestId = usize;

pub struct ReadRequest {
    pub(crate) id: RequestId,
    pub(crate) offset: u64,
    pub(crate) length: usize,
    pub(crate) alignment: Alignment,
    pub(crate) callback: oneshot::Sender<VortexResult<BufferHandle>>,
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
    pub(crate) fn resolve(self, result: VortexResult<BufferHandle>) {
        let result = result.and_then(|buffer| {
            if self.length != buffer.len() {
                vortex_bail!(
                    "ReadRequest: expected buffer of length {} but received {}.",
                    self.length,
                    buffer.len()
                )
            }
            Ok(buffer)
        });

        if let Err(e) = self.callback.send(result) {
            tracing::debug!("ReadRequest {} dropped before resolving: {e}", self.id);
        }
    }
}

/// A set of I/O requests that have been coalesced into a single larger request.
pub(crate) struct CoalescedRequest {
    pub(crate) range: Range<u64>,
    pub(crate) alignment: Alignment, // Global max segment alignment used for the coalesced range.
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
    pub fn resolve(self, result: VortexResult<BufferHandle>) {
        let result = result.and_then(|buffer| {
            let expected_length = self.range.end.saturating_sub(self.range.start);
            let buffer_len = buffer.len() as u64;
            if expected_length != buffer_len {
                vortex_bail!(
                    "CoalescedRequest: expected buffer of length {} but received {}.",
                    expected_length,
                    buffer_len
                )
            }

            for req in self.requests.iter() {
                let request_offset = req.offset.checked_sub(self.range.start).ok_or_else(|| {
                    vortex_err!(
                        "CoalescedRequest: sub-request for length {} at file offset {} preceeds coalesced range: {}..{}.",
                        req.length,
                        req.offset,
                        self.range.start,
                        self.range.end,
                    )
                })?;
                if request_offset > buffer_len {
                    vortex_bail!(
                        "CoalescedRequest: sub-request for length {} at buffer offset {} (file offset {}) is unsatisfiable by buffer of length {}.",
                        req.length,
                        request_offset,
                        req.offset,
                        buffer_len
                    )
                }
                let request_end = request_offset.saturating_add(req.length as u64);
                if request_end > buffer_len {
                    vortex_bail!(
                        "CoalescedRequest: sub-request for length {} at buffer offset {} (file offset {}) is unsatisfiable by buffer of length {}.",
                        req.length,
                        request_offset,
                        req.offset,
                        buffer_len
                    )
                }
            }
            Ok(buffer)
        });

        match result {
            Ok(buffer) => {
                let base = match buffer.ensure_aligned(Alignment::none()) {
                    Ok(base) => base,
                    Err(e) => {
                        let e = Arc::new(e);
                        for req in self.requests.into_iter() {
                            req.resolve(Err(VortexError::from(Arc::clone(&e))));
                        }
                        return;
                    }
                };

                for req in self.requests.into_iter() {
                    let start = usize::try_from(req.offset - self.range.start)
                        .vortex_expect("invalid offset");
                    let end = start + req.length;
                    let slice = match base.slice(start..end).ensure_aligned(req.alignment) {
                        Ok(slice) => slice,
                        Err(e) => {
                            req.resolve(Err(e));
                            continue;
                        }
                    };
                    req.resolve(Ok(slice));
                }
            }
            Err(e) => {
                let e = Arc::new(e);
                for req in self.requests.into_iter() {
                    req.resolve(Err(VortexError::from(Arc::clone(&e))));
                }
            }
        }
    }
}
