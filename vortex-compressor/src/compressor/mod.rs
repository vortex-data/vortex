// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod sample;
mod select;
mod structural;

use std::sync::Arc;

use crate::builtins::IntDictScheme;
use crate::scheme::AllowedSerializedIds;
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

    /// The serialized IDs the writer may emit, handed to every [`CompressorContext`].
    allowed_serialized_ids: Arc<AllowedSerializedIds>,
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
            allowed_serialized_ids: Arc::new(AllowedSerializedIds::All),
        }
    }

    /// Hands the compressor the serialized IDs the writer may emit, intersecting with any
    /// earlier call.
    ///
    /// The set reaches every scheme through [`CompressorContext::allows_serialized_id`], so a
    /// scheme with several wire formats writes a newer one only when permitted. Callers filter
    /// the scheme list themselves: every scheme given to [`new`](Self::new) should have all of its
    /// [`produced_encodings`](Scheme::produced_encodings) permitted, as
    /// `BtrBlocksCompressorBuilder::retain_allowed_encodings` ensures.
    pub fn with_allowed_serialized_ids(mut self, allowed: &AllowedSerializedIds) -> Self {
        let mut merged = (*self.allowed_serialized_ids).clone();
        merged.intersect(allowed);
        self.allowed_serialized_ids = Arc::new(merged);
        self
    }

    /// The serialized IDs the writer may emit.
    pub fn allowed_serialized_ids(&self) -> &AllowedSerializedIds {
        &self.allowed_serialized_ids
    }

    /// The compression schemes, in registration order.
    pub fn schemes(&self) -> &[&'static dyn Scheme] {
        &self.schemes
    }

    /// The context a compress call starts from, carrying the permitted serialized IDs.
    pub(crate) fn root_context(&self) -> CompressorContext {
        CompressorContext::new()
            .with_allowed_serialized_ids(Arc::clone(&self.allowed_serialized_ids))
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

#[cfg(test)]
mod edition_tests;
