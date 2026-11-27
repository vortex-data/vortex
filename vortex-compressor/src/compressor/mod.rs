// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod encoded;
mod sample;
mod select;
mod structural;

use std::sync::Arc;

pub(crate) use cascade::TrialOutput;
pub(crate) use cascade::exact_uncompressed_size;
pub(crate) use sample::estimate_compression_ratio_with_sampling;
pub use sample::sample_aligned_chunks;
pub(crate) use sample::sample_slices;

use crate::builtins::IntDictScheme;
use crate::scheme::ChildPolicy;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// Synthetic scheme ID used for the compressor's own root-level cascading.
pub(crate) const ROOT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.root",
};

/// The main compressor type implementing cascading adaptive compression.
///
/// This compressor applies adaptive compression [`Scheme`]s to arrays based on their data types and
/// characteristics. It recursively compresses nested structures like structs and lists, and chooses
/// optimal compression schemes for leaf types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation.
/// 2. Pre-filtering schemes by [`Scheme::matches`] and exclusion rules.
/// 3. Evaluating each matching scheme's compression estimate and resolving deferred work.
/// 4. Encoding with the best scheme, compressing the children it registers, and verifying the
///    result is smaller.
///
/// No scheme may appear twice in a cascade chain. The compressor enforces this automatically
/// along with push/pull exclusion rules declared by each scheme.
///
/// Downstream crates usually wrap this type with a preconfigured scheme set. Use it directly when
/// embedding a custom fixed scheme list or testing scheme interactions.
#[derive(Debug, Clone)]
pub struct CascadingCompressor {
    /// The enabled compression schemes.
    schemes: Vec<&'static dyn Scheme>,

    /// Descendant exclusion rules for the compressor's own cascading (e.g. excluding Dict from
    /// list offsets).
    root_exclusions: Vec<DescendantExclusion>,

    /// Overrides schemes' decisions about which children to compress.
    child_policy: Option<Arc<dyn ChildPolicy>>,

    /// Compresses the storage children of Variant arrays in place of this compressor, if set.
    variant_storage: Option<Arc<CascadingCompressor>>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built automatically.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(structural::root_list_children::OFFSETS),
        }];

        Self {
            schemes,
            root_exclusions,
            child_policy: None,
            variant_storage: None,
        }
    }

    /// Sets a policy that decides, child by child, whether to compress what schemes register,
    /// overriding the schemes' own decisions.
    pub fn with_child_policy(mut self, policy: Arc<dyn ChildPolicy>) -> Self {
        self.child_policy = Some(policy);
        self
    }

    /// Compresses the storage children of Variant arrays with `compressor` instead of this one.
    ///
    /// The storage of a Variant holds opaque binary values, such as its unshredded value, which
    /// can warrant other schemes than the binary columns of a table. Shredded children are still
    /// compressed by this compressor.
    pub fn with_variant_storage_compressor(mut self, compressor: CascadingCompressor) -> Self {
        self.variant_storage = Some(Arc::new(compressor));
        self
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;
