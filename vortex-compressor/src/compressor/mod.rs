// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod sample;
mod select;
mod structural;

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_map::HashMap;
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

    /// Maps every registered version to the version selected for compression.
    scheme_aliases: HashMap<SchemeId, SchemeId>,

    /// Configuration only: retained so repeated restrictions intersect exactly.
    allowed_serialized_ids: Option<HashSet<ArrayId>>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// Register only the newest version of each scheme. Predecessor IDs are aliases for the
    /// selected version in exclusions and [`has_scheme`](Self::has_scheme) checks.
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built automatically.
    ///
    /// # Panics
    ///
    /// Panics if predecessor chains contain a cycle or share a scheme ID, including when multiple
    /// versions of the same scheme are registered separately.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        let mut scheme_aliases = HashMap::new();
        for &scheme in &schemes {
            let mut candidate = Some(scheme);
            while let Some(version) = candidate {
                assert!(
                    scheme_aliases.insert(version.id(), scheme.id()).is_none(),
                    "scheme {} appears more than once in the registered predecessor chains",
                    version.id(),
                );
                candidate = version.predecessor();
            }
        }

        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(structural::root_list_children::OFFSETS),
        }];

        Self {
            schemes,
            root_exclusions,
            scheme_aliases,
            allowed_serialized_ids: None,
        }
    }

    /// Selects the newest eligible version of each scheme, intersecting with any earlier call.
    ///
    /// A version is eligible only when all of its [`Scheme::required_serialized_ids`] are allowed.
    /// Otherwise its predecessors are tried in order; the scheme is removed if none is eligible.
    /// Selection preserves registration order and happens before any compression or estimation.
    pub fn with_allowed_serialized_ids(mut self, allowed: HashSet<ArrayId>) -> Self {
        let allowed = match self.allowed_serialized_ids.take() {
            Some(existing) => existing.intersection(&allowed).copied().collect(),
            None => allowed,
        };
        let mut replacements = HashMap::new();
        self.schemes = self
            .schemes
            .into_iter()
            .filter_map(|scheme| {
                let mut candidate = Some(scheme);
                while let Some(version) = candidate {
                    if version
                        .produced_encodings()
                        .iter()
                        .all(|id| allowed.contains(id))
                    {
                        replacements.insert(scheme.id(), version.id());
                        return Some(version);
                    }
                    candidate = version.predecessor();
                }
                None
            })
            .collect();
        self.scheme_aliases.retain(|_, selected| {
            if let Some(replacement) = replacements.get(selected) {
                *selected = *replacement;
                true
            } else {
                false
            }
        });
        self.allowed_serialized_ids = Some(allowed);
        self
    }

    /// The context a compress call starts from.
    pub(crate) fn root_context(&self) -> CompressorContext {
        CompressorContext::new()
    }

    /// Returns whether a version of `scheme` is enabled.
    ///
    /// Any ID in a registered predecessor chain refers to the selected version, including when
    /// the selected version is older or newer than the specified ID.
    pub fn has_scheme(&self, scheme: SchemeId) -> bool {
        self.scheme_aliases.contains_key(&scheme)
    }

    fn resolve_scheme_id(&self, scheme: SchemeId) -> SchemeId {
        self.scheme_aliases.get(&scheme).copied().unwrap_or(scheme)
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;

#[cfg(test)]
mod version_tests;
