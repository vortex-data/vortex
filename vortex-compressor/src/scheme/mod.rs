// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Everything a scheme author implements or receives: the [`Scheme`] trait, exclusion rules,
//! compression estimates, and the compression context.

mod ctx;
pub use ctx::CompressorContext;
pub use ctx::MAX_CASCADE;

pub(crate) mod evaluation;
mod exclusion;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

pub use evaluation::CandidateEstimate;
pub use evaluation::DeferredEvaluation;
pub use evaluation::DeferredEvaluationFn;
pub use evaluation::ResolvedEvaluation;
pub use evaluation::SchemeEvaluation;
pub use exclusion::AncestorExclusion;
pub use exclusion::ChildSelection;
pub use exclusion::DescendantExclusion;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Unique identifier for a compression scheme.
///
/// The only way to obtain a [`SchemeId`] is through [`SchemeExt::id()`], which is auto-implemented
/// for all [`Scheme`] types. There is no public constructor.
///
/// The only exception to this is for the compressor's synthetic `ROOT_SCHEME_ID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemeId {
    /// Only constructable within `vortex-compressor`.
    ///
    /// The only public way to obtain a [`SchemeId`] is through [`SchemeExt::id()`].
    pub(super) name: &'static str,
}

impl fmt::Display for SchemeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

// TODO(connor): Remove all default implemented methods.
/// A single compression encoding that the [`CascadingCompressor`] can select from.
///
/// The compressor evaluates every registered scheme whose [`matches`] returns `true` for a given
/// array, asks the configured [`CostModel`] to compute each candidate's cost, and calls [`compress`]
/// on the lowest-cost eligible candidate. The default [`SizeCost`] model preserves the historical
/// highest-compression-ratio selection.
///
/// One of the key features of the compressor in this crate is that schemes may "cascade". A
/// scheme's [`compress`] can call back into the compressor via
/// [`CascadingCompressor::compress_child`] to compress child or transformed arrays, building up
/// multiple encoding layers (e.g. frame-of-reference and then bit-packing).
///
/// # Scheme IDs
///
/// Every scheme has a globally unique name returned by [`scheme_name`]. The [`SchemeExt::id`]
/// method (auto-implemented, cannot be overridden) wraps that name in an opaque [`SchemeId`] used
/// for equality, hashing, and exclusion rules (see below).
///
/// # Cascading and children
///
/// Schemes that produce child arrays for further compression must declare [`num_children`] > 0.
/// Each child should be identified by a stable index. Cascading schemes should use
/// [`CascadingCompressor::compress_child`] to compress each child array, which handles cascade
/// level / budget tracking and context management automatically.
///
/// No scheme may appear twice in a cascade (descendant) chain (enforced by the compressor). This
/// keeps the search space a tree.
///
/// # Exclusion rules
///
/// Schemes declare exclusion rules to prevent incompatible scheme combinations in the cascade
/// chain:
///
/// - [`descendant_exclusions`] (push): "exclude scheme X from my child Y's subtree." Used when the
///   declaring scheme knows about the excluded scheme.
/// - [`ancestor_exclusions`] (pull): "exclude me if ancestor X's child Y is above me." Used when
///   the declaring scheme knows about the ancestor.
///
/// We do this because different schemes will live in different crates, and we cannot know the
/// dependency direction ahead of time.
///
/// # Implementing a scheme
///
/// [`evaluate`] should return `SchemeEvaluation::Deferred(DeferredEvaluation::Sample)` when a
/// candidate cannot be described cheaply, asking the compressor to evaluate it through sampling.
/// Implementors should return an immediate [`SchemeEvaluation::Candidate`] when possible.
///
/// Schemes that need statistics that may be expensive to compute should override [`stats_options`]
/// to declare what they require. The compressor merges all eligible schemes' options before
/// generating stats, so each stat is always computed at most once for a given array.
///
/// A scheme implementation should be deterministic for a fixed input array and context. The
/// compressor uses scheme order for deterministic tie-breaking, so non-deterministic estimates make
/// compressed output harder to reproduce and compare.
///
/// [`scheme_name`]: Scheme::scheme_name
/// [`matches`]: Scheme::matches
/// [`compress`]: Scheme::compress
/// [`evaluate`]: Scheme::evaluate
/// [`CostModel`]: crate::cost::CostModel
/// [`SizeCost`]: crate::cost::SizeCost
/// [`stats_options`]: Scheme::stats_options
/// [`num_children`]: Scheme::num_children
/// [`descendant_exclusions`]: Scheme::descendant_exclusions
/// [`ancestor_exclusions`]: Scheme::ancestor_exclusions
pub trait Scheme: Debug + Send + Sync {
    /// The globally unique name for this scheme (e.g. `"vortex.int.bitpacking"`).
    fn scheme_name(&self) -> &'static str;

    /// Whether this scheme can compress the given canonical array.
    fn matches(&self, canonical: &Canonical) -> bool;

    /// Returns the stats generation options this scheme requires. The compressor merges all
    /// eligible schemes' options before generating stats so that a single stats pass satisfies
    /// every scheme.
    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions::default()
    }

    /// The number of child arrays this scheme produces when cascading. Returns 0 for leaf
    /// schemes that produce a final encoded array.
    fn num_children(&self) -> usize {
        0
    }

    /// Schemes to exclude from specific children's subtrees (push direction).
    ///
    /// Each rule says: "when I cascade through child Y, do not use scheme X anywhere in that
    /// subtree." Only meaningful when [`num_children`](Scheme::num_children) > 0.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        Vec::new()
    }

    /// Ancestors that make this scheme ineligible (pull direction).
    ///
    /// Each rule says: "if ancestor X cascaded through child Y somewhere above me in the chain, do
    /// not try me."
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        Vec::new()
    }

    /// Produces model-independent candidate evidence for this scheme on the given array.
    ///
    /// This method should be fast and infallible. Any expensive or fallible work should be
    /// deferred to the compressor by returning
    /// `SchemeEvaluation::Deferred(DeferredEvaluation::Sample)` or
    /// `SchemeEvaluation::Deferred(DeferredEvaluation::Callback(...))`.
    ///
    /// The compressor combines the returned estimate with compressor-owned selection context to
    /// construct a [`crate::cost::Candidate`], then asks the configured
    /// [`crate::cost::CostModel`] to compute its cost. The default [`crate::cost::SizeCost`] model
    /// interprets the candidate's estimated compression ratio; other models may use the remaining
    /// candidate evidence differently.
    ///
    /// [`SchemeEvaluation::Candidate`] means the scheme can immediately describe a candidate.
    /// `SchemeEvaluation::Deferred(DeferredEvaluation::Sample)` asks the compressor to sample;
    /// `SchemeEvaluation::Deferred(DeferredEvaluation::Callback(...))` asks it to run custom
    /// deferred work. Deferred callbacks must return a terminal
    /// [`ResolvedEvaluation`], never another deferred request.
    ///
    /// Note that the compressor will also use this method when compressing samples, so some
    /// statistics that might hold for the samples may not hold for the entire array (e.g.,
    /// constancy). Implementations should check `ctx.is_sample` to make sure that they are
    /// returning the correct information.
    ///
    /// The compressor guarantees that empty and all-null arrays are handled before this method is
    /// called, so implementations may assume the array has at least one valid element. Outside of
    /// sample compression, the compressor also encodes constant arrays itself before evaluating
    /// schemes, so implementations only see constant arrays when `ctx.is_sample()` is `true`.
    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation;

    /// Compress the array using this scheme.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;
}

impl PartialEq for dyn Scheme {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn Scheme {}

impl Hash for dyn Scheme {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

/// Extension trait providing [`id`](SchemeExt::id) for all [`Scheme`] implementors.
///
/// This trait is automatically implemented for every type that implements [`Scheme`]. Because the
/// blanket implementation covers all types, external crates cannot override `id()`.
pub trait SchemeExt: Scheme {
    /// Unique identifier derived from [`scheme_name`](Scheme::scheme_name).
    fn id(&self) -> SchemeId {
        SchemeId {
            name: self.scheme_name(),
        }
    }
}

impl<T: Scheme + ?Sized> SchemeExt for T {}
