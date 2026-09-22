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

    /// The serialized IDs the compressor may emit. See [`Self::with_allowed_serialized_ids`].
    allowed_serialized_ids: Arc<HashSet<ArrayId>>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built automatically.
    /// The compressor may emit every serialized ID its schemes declare in
    /// [`Scheme::produced_encodings`], and nothing else, until
    /// [`with_allowed_serialized_ids`](Self::with_allowed_serialized_ids) says otherwise.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(structural::root_list_children::OFFSETS),
        }];
        let allowed_serialized_ids = schemes
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();

        Self {
            schemes,
            root_exclusions,
            allowed_serialized_ids: Arc::new(allowed_serialized_ids),
        }
    }

    /// Restricts the compressor to the serialized IDs in `allowed`.
    ///
    /// Schemes declaring an ID outside `allowed` are removed. The remaining schemes see `allowed`
    /// through [`allows_serialized_id`](crate::scheme::CompressorContext::allows_serialized_id),
    /// which lets a scheme emit an optional wire format only when the writer permits it. The file
    /// writer passes the serialized IDs its enabled editions permit.
    pub fn with_allowed_serialized_ids(mut self, allowed: HashSet<ArrayId>) -> Self {
        self.schemes.retain(|scheme| {
            scheme
                .produced_encodings()
                .iter()
                .all(|id| allowed.contains(id))
        });
        self.allowed_serialized_ids = Arc::new(allowed);
        self
    }

    /// The serialized IDs this compressor may emit.
    pub fn allowed_serialized_ids(&self) -> &HashSet<ArrayId> {
        &self.allowed_serialized_ids
    }

    /// Whether a scheme with the given ID is registered.
    pub fn has_scheme(&self, id: SchemeId) -> bool {
        self.schemes.iter().any(|scheme| scheme.id() == id)
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;
