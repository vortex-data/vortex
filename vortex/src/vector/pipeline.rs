// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::Vector;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// A pipeline provides a push-based way to emit a stream of vectors.
///
/// By passing multiple vector computations through the same pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
///
/// I haven't yet figured out the exact semantics of a pipeline. There's a few high-level options:
/// * We pass a mask into `next` and expect a vector containing exactly the true count.
/// * We pass a mask into `next` and expect a vector containing less than or equal to the true count.
/// * We do not pass a mask into `next` and expect a vector containing whatever is easy for the
///   pipeline to export.
///
/// By allowing pipelines to return partial results, we enable them to throw away their leading
/// data that may not be exactly aligned with the ideal `N` elements per vector. However, if
/// we allow this, then it's hardly likely that join nodes in the pipeline will line up, and so we
/// will need complex buffering logic.
///
/// By passing a mask into `next`, we do provide visibility into the density of the data. Some
/// arrays may choose to decompress the full `N` elements, and then use the given mask as the
/// vector's selection mask. If two vectors do the same, we can compare the masks and then perform
/// an operation without needing to perform the selection. If the selection masks are not the same,
/// we need a way to flatten a vector into a "prefix" vector, where the first `true_count` elements
/// are dense at the start of the vector, allowing us to still use vectorized kernels. If a
/// selection mask is _too_ sparse, e.g. a handful of elements, then we may wish to buffer adjacent
/// masks together to form a denser vector. Ideally, the compute function that instantiate the
/// pipeline would have full visibility into the larger mask and can make that decision.
///
/// There may be value in allowing pipelines to export no data. In other words, the `next` function
/// indicates that the data isn't ready yet (ideally it has populated some sort of context in
/// the meantime that allows the caller to make progress, for example, downloading segments). But
/// these feels like future work where we potentially merge arrays and layouts.
///
// TODO(ngates): we should explore a version of pipelines that are not object-safe traits. This
//  would allow for compile-time optimizations and inlining for the cases where we wish to
//  "semi-fuse" kernels.
pub trait Pipeline {
    /// Exports the next vector from the pipeline, given a length [`N`] mask and an output vector.
    ///
    /// Note that while the input mask is a bit array, the output vector's selection mask is in
    /// terms of indices. This is because we can then read elements using `elems[selection[idx]]`
    /// whereas a bit-mask would require much slower ranked selection.
    /// Perhaps not with this? Or with a SIMD unpack?
    ///  <https://www.microsoft.com/en-us/research/publication/selection-pushdown-in-column-stores-using-bit-manipulation-instructions/>
    fn next<'v>(&mut self, mask: &Mask, out: &'v mut Vector<'v>) -> VortexResult<()>;
}

pub trait SupportsPipeline {
    /// Returns a pipeline that can be used to export canonical data from this array.
    fn pipeline(&self) -> Box<dyn Pipeline>;

    // TODO(ngates): there will be another function, similar to find_kernel, that takes a compute
    //  function and returns a pipeline?
}
