// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified compression scheme trait and exclusion rules.

use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Unique identifier for a compression scheme.
///
/// The only way to obtain a [`SchemeId`] is through [`SchemeExt::id()`], which is
/// auto-implemented for all [`Scheme`] types. There is no public constructor.
///
/// The only exception to this is for `ROOT_SCHEME_ID` in `compressor.rs`.
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

/// Selects which children of a cascading scheme a rule applies to.
#[derive(Debug, Clone, Copy)]
pub enum ChildSelection {
    /// Rule applies to all children.
    All,
    /// Rule applies to a single child.
    One(usize),
    /// Rule applies to multiple specific children.
    Many(&'static [usize]),
}

impl ChildSelection {
    /// Returns `true` if this selection includes the given child index.
    pub fn contains(&self, child_index: usize) -> bool {
        match self {
            ChildSelection::All => true,
            ChildSelection::One(idx) => *idx == child_index,
            ChildSelection::Many(indices) => indices.contains(&child_index),
        }
    }
}

/// Push rule: declared by a cascading scheme to exclude another scheme from the subtree
/// rooted at the specified children.
///
/// Use this when the declaring scheme (the ancestor) knows about the excluded scheme. For example,
/// `ZigZag` excludes `Dict` from all its children.
#[derive(Debug, Clone, Copy)]
pub struct DescendantExclusion {
    /// The scheme to exclude from descendants.
    pub excluded: SchemeId,
    /// Which children of the declaring scheme this rule applies to.
    pub children: ChildSelection,
}

/// Pull rule: declared by a scheme to exclude itself when the specified ancestor is in the
/// cascade chain.
///
/// Use this when the excluded scheme (the descendant) knows about the ancestor. For example,
/// `Sequence` excludes itself when `IntDict` is an ancestor on its codes child.
#[derive(Debug, Clone, Copy)]
pub struct AncestorExclusion {
    /// The ancestor scheme that makes the declaring scheme ineligible.
    pub ancestor: SchemeId,
    /// Which children of the ancestor this rule applies to.
    pub children: ChildSelection,
}

// TODO(connor): Remove all default implemented methods.
/// A single compression encoding that the [`CascadingCompressor`] can select from.
///
/// The compressor evaluates every registered scheme whose [`matches`] returns `true` for a given
/// array, picks the one with the highest [`expected_compression_ratio`], and calls [`compress`] on
/// the winner.
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
/// [`expected_compression_ratio`] should return
/// `CompressionEstimate::Deferred(DeferredEstimate::Sample)` when a cheap heuristic is not
/// available, asking the compressor to estimate via sampling. Implementors should return an
/// immediate [`CompressionEstimate::Verdict`] when possible.
///
/// Schemes that need statistics that may be expensive to compute should override [`stats_options`]
/// to declare what they require. The compressor merges all eligible schemes' options before
/// generating stats, so each stat is always computed at most once for a given array.
///
/// [`scheme_name`]: Scheme::scheme_name
/// [`matches`]: Scheme::matches
/// [`compress`]: Scheme::compress
/// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
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

    /// Cheaply estimate the compression ratio for this scheme on the given array.
    ///
    /// This method should be fast and infallible. Any expensive or fallible work should be
    /// deferred to the compressor by returning
    /// `CompressionEstimate::Deferred(DeferredEstimate::Sample)` or
    /// `CompressionEstimate::Deferred(DeferredEstimate::Callback(...))`.
    ///
    /// The compressor will ask all schemes what their expected compression ratio is given the array
    /// and statistics. The scheme with the highest estimated ratio will then be applied to the
    /// entire array.
    ///
    /// [`CompressionEstimate::Verdict`] means the scheme already knows the terminal
    /// [`crate::estimate::EstimateVerdict`]. `CompressionEstimate::Deferred(DeferredEstimate::Sample)`
    /// asks the compressor to sample. `CompressionEstimate::Deferred(DeferredEstimate::Callback(...))`
    /// asks the compressor to run custom deferred work. Deferred callbacks must return a
    /// [`crate::estimate::EstimateVerdict`] directly, never another deferred request.
    ///
    /// Note that the compressor will also use this method when compressing samples, so some
    /// statistics that might hold for the samples may not hold for the entire array (e.g.,
    /// `Constant`). Implementations should check `ctx.is_sample` to make sure that they are
    /// returning the correct information.
    ///
    /// The compressor guarantees that empty and all-null arrays are handled before this method is
    /// called. Implementations may assume the array has at least one valid element. However, a
    /// constant scheme should still be registered with the compressor to detect single-value arrays
    /// that are not all-null.
    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate;

    /// Compress the array using this scheme.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
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
