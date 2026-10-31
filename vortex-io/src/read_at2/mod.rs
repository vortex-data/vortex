// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Completion-based positional reads. Each reader is an independent registration scope with
//! one completion consumer. Implementations may use futures or an externally driven reactor.
//! Request identifiers must not be reused within a scope, including after cancellation.

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

mod adapter;
mod service;
#[cfg(test)]
mod tests;

pub use adapter::FutureReadAt2;
pub use service::ReadAt2IoService;

/// Identifies a request within one reader scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadId(pub u64);

/// The information requested from a byte source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadTarget {
    /// Source length in bytes.
    Size,
    /// Exactly `length` bytes at an absolute offset.
    Range {
        /// Absolute offset.
        offset: u64,
        /// Required length.
        length: usize,
        /// Required alignment of the delivered buffer.
        alignment: Alignment,
    },
    /// Up to `length` trailing bytes, together with the source size.
    Suffix {
        /// Maximum number of trailing bytes.
        length: usize,
        /// Required alignment of the delivered buffer.
        alignment: Alignment,
    },
}

/// A positional request with caller-assigned identity.
#[derive(Clone, Copy, Debug)]
pub struct ReadRequest {
    /// Unique identity within the reader scope.
    pub id: ReadId,
    /// Requested information.
    pub target: ReadTarget,
}

/// Result corresponding to a [`ReadTarget`].
#[derive(Debug)]
pub enum ReadResult {
    /// Source length.
    Size(u64),
    /// Exact range, preserving host or device residency.
    Bytes(BufferHandle),
    /// Source length and trailing bytes.
    Suffix {
        /// Source length.
        size: u64,
        /// Trailing bytes.
        bytes: BufferHandle,
    },
}

/// One completed request, including failures.
pub struct ReadCompletion {
    /// Identity of the completed request.
    pub id: ReadId,
    /// Result of the operation.
    pub result: VortexResult<ReadResult>,
}

/// An independent completion queue for positional reads of one source.
///
/// Submission only registers work. Polling may advance the implementation's driver without
/// blocking. A pending asynchronous poll must register its waker. Reactor-backed readers require
/// their owning reactor to keep running; registering a waker does not drive that reactor.
pub trait VortexReadAt2: Send + Sync + 'static {
    /// Registers a batch atomically; a registration error accepts none of it.
    fn submit(&self, requests: &[ReadRequest]) -> VortexResult<()>;

    /// Takes an immediately available result without advancing or submitting queued IO.
    /// Readers may leave this optimization unsupported and return completions through polling.
    fn take_ready(&self) -> VortexResult<Option<ReadCompletion>> {
        Ok(None)
    }

    /// Consumes an available completion without waiting; may advance the implementation's driver.
    fn try_complete(&self) -> VortexResult<Option<ReadCompletion>>;

    /// Consumes a completion, registering a wakeup before returning pending.
    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>>;

    /// Withdraws interest, suppressing delivery. In-flight resources remain owned by the backend.
    fn cancel(&self, id: ReadId);

    /// Drives a blocking completion wait when the implementation supports native waiting.
    /// Browser and executor-only readers may leave this capability unsupported.
    fn wait(&self) -> VortexResult<ReadCompletion> {
        vortex_bail!("This reader requires asynchronous completion polling")
    }
}

/// Creates an independent reader scope on the thread that will drive it.
pub type ReadAt2Factory = Arc<dyn Fn() -> VortexResult<Arc<dyn VortexReadAt2>> + Send + Sync>;
