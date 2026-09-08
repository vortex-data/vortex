// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod sample;
mod select;
mod structural;

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_set::HashSet;

use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressorContext;
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
/// 4. Compressing with the best scheme and verifying the result is smaller.
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

    /// The serialized IDs the writer may emit, or `None` for no restriction. Seeds every root
    /// [`CompressorContext`], where schemes read it.
    allowed_serialized_ids: Option<Arc<HashSet<ArrayId>>>,
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
            allowed_serialized_ids: None,
        }
    }

    /// Hands the compressor the serialized IDs the writer may emit, intersecting with any earlier
    /// call.
    ///
    /// The file writer passes the serialized IDs its enabled editions permit. Schemes read the
    /// set through [`CompressorContext::allows_serialized_id`], so a scheme whose encoding has
    /// several wire formats picks the newest permitted one as its mode, while estimating and
    /// while compressing alike.
    pub fn with_allowed_serialized_ids(mut self, allowed: HashSet<ArrayId>) -> Self {
        self.allowed_serialized_ids = Some(Arc::new(match self.allowed_serialized_ids.take() {
            Some(existing) => existing.intersection(&allowed).copied().collect(),
            None => allowed,
        }));
        self
    }

    /// The serialized IDs the writer may emit, or `None` when unrestricted.
    pub fn allowed_serialized_ids(&self) -> Option<&HashSet<ArrayId>> {
        self.allowed_serialized_ids.as_deref()
    }

    /// The context a compress call starts from.
    pub(crate) fn root_context(&self) -> CompressorContext {
        CompressorContext::new(self.allowed_serialized_ids.clone())
    }

    /// Returns whether the compressor was configured with `scheme`.
    pub fn has_scheme(&self, scheme: SchemeId) -> bool {
        self.schemes
            .iter()
            .any(|candidate| candidate.id() == scheme)
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;
