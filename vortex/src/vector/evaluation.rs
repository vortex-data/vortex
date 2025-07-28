// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::exporter::Exporter;
use vortex_array::Canonical;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// An evaluation provides a push-based way to emit a stream of vectors.
///
/// Should we rename this to `Pipeline`?
///
/// By passing multiple vector computations through the same evaluation pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.). It is also possible
/// to construct and reuse cached nodes in the evaluation graph, for example, creating a `tee`
/// node to emit the same data to multiple exporters and avoid duplicate computation.
///
/// Passing in an `Exporter` (instead of say `&mut dyn Array`) allows us to have more explicit
/// control over what happens if the evaluation wants to return a non-canonical encoding.
/// into it.
///
/// FIXME(ngates): this has a similar problem to Layouts. It would be useful if the array had full
///  visibility into the mask it is about to be asked for, so it can optimize its export? Actually,
///  when is this true? I guess if there is a series of low selectivity masks, then it would be
///  useful to export these scalars into the same repartitioned vector. Also, does each invocation
///  assume the mask has been consumed? Or can the implementation return the number of rows it has
///  consumed from the mask? This could be useful for FastLanes where if it's given a 4k mask, it
///  could just look at the first 1k, export that, then return the fact that it only consumed 1k.
///  This may fuck with alignment though...?
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
    ///
    /// By allowing the implementation to export less than the mask's true count, chunked arrays
    /// are able to align themselves with the export graph. For example, a 1k chunked array could
    /// emit some leading rows, before emitting subsequent rows as aligned 1k chunks.
    ///
    /// Implementations are expected to either export data into the pre-allocated canonical vector
    /// contained within the `Exporter`, or by calling `Exporter::export` with an arbitrarily
    /// encoded vector. Callers may choose how to handle the latter case (for example, by
    /// canonicalizing the vector to use it, or by propagating it in some way).
    ///
    /// FIXME(ngates): what if the evaluation pipeline depends on the exported vector's encoding???
    fn next(&mut self, mask: &Mask, out: Exporter) -> VortexResult<()>;
}

impl dyn Evaluation + '_ {
    /// It should be possible to collect an evaluation into a canonical Array by exporting to
    /// consecutive positions in a pre-allocated output array.
    pub fn collect(self) -> Canonical {
        todo!()
    }
}
