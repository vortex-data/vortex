// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod sample;
mod select;
mod structural;

use std::collections::BTreeMap;
use std::sync::Arc;

use vortex_array::ArrayId;

use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// The maximum compatible writer version enabled for each array encoding.
///
/// This is a write-time policy. It is consulted before a scheme is estimated or run and is never
/// stored in an array or used to select a reader.
pub type ArrayWriterVersions = BTreeMap<ArrayId, u16>;

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

    /// Per-array writer ceilings, when compression is constrained for serialization.
    array_writer_versions: Option<Arc<ArrayWriterVersions>>,
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
            array_writer_versions: None,
        }
    }

    /// Constrains schemes to serialized features allowed by these per-array writer versions.
    ///
    /// Schemes whose required version is absent or newer than the configured ceiling are removed
    /// before statistics, estimation, sampling, or compression. Without this policy, compression
    /// is intended for in-memory use and all registered scheme versions remain eligible.
    pub fn with_array_writer_versions(mut self, versions: ArrayWriterVersions) -> Self {
        self.array_writer_versions = Some(Arc::new(versions));
        self
    }

    /// Whether `scheme` can produce its representation of `canonical` under the writer policy.
    fn writer_version_allows(
        &self,
        scheme: &dyn Scheme,
        canonical: &vortex_array::Canonical,
    ) -> bool {
        let Some(enabled) = &self.array_writer_versions else {
            return true;
        };

        scheme
            .required_array_writer_versions(canonical)
            .into_iter()
            .all(|(id, required)| enabled.get(&id).is_some_and(|enabled| *enabled >= required))
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;
