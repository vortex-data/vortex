// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::exporter::Exporter;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// An evaluation provides a push-based way to emit a stream of vectors.
///
/// By passing multiple vector computations through the same evaluation pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.) up-front.
pub trait Evaluation {
    /// The `next` function is called to export the next batch of data into the provided `Exporter`.
    ///
    /// This function should be called repeatedly until the expected number of rows has been
    /// returned. Intermediate calls *may emit empty vectors*. This allows implementations to
    /// delay compute until some external condition is met, such as I/O completion.
    ///
    /// The data exported to `out` will be less than or equal to the [`Mask::true_count`] of the
    /// provided `mask` during each invocation. After all invocations, the total number of rows
    /// exported must equal the total number of `true` values in the global mask.
    fn next(&mut self, mask: &Mask, out: Exporter) -> VortexResult<()>;
}
