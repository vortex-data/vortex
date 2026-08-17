// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The result of a completed row loop before batch-level null handling.
//!
//! [`RowExecution`] preserves deferred failure evidence until batch execution can determine whether
//! the failing payload belonged to a valid row.

use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;

/// The outcome of a row loop before batch execution decides whether an error is observable.
///
/// Together with the surrounding [`VortexResult`], this represents three outcomes:
///
/// - `Err(error)` is a non-retryable execution or immediate row error.
/// - [`Output`](Self::Output) is a successful row loop.
/// - [`DeferredError`](Self::DeferredError) is failure evidence from a completed row loop.
///
/// A dense loop can evaluate null payloads, so its deferred error is not always observable. Batch
/// execution can retry only valid rows to discard errors caused by null payloads. A plain
/// `VortexResult<ArrayRef>` cannot distinguish these errors from failures that a retry cannot fix.
///
/// Once execution is known to contain only valid rows, converting this outcome into a
/// `VortexResult<ArrayRef>` turns [`DeferredError`](Self::DeferredError) into an ordinary error.
pub enum RowExecution {
    /// The successfully built, full-length output column.
    Output(ArrayRef),

    /// An error constructed from failure evidence reduced across a completed row loop.
    DeferredError(VortexError),
}

impl From<RowExecution> for VortexResult<ArrayRef> {
    fn from(execution: RowExecution) -> Self {
        match execution {
            RowExecution::Output(output) => Ok(output),
            RowExecution::DeferredError(error) => Err(error),
        }
    }
}
