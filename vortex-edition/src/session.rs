// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`EditionSession`] session variable: the per-session registry of editions and
//! edition inclusions.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionError;
use crate::EditionId;
use crate::EditionInclusion;
use crate::parse_release;

/// The session's registry of editions and edition inclusions.
///
/// Starts empty and is populated at initialization time: the `vortex` facade seeds the
/// first-party declarations (`vortex::editions`), and crates registering additional
/// encodings declare their inclusions (and, for new families, editions) alongside their
/// encoding registration. Clones share the same underlying registry, matching the other
/// session registries.
#[derive(Clone, Debug, Default)]
pub struct EditionSession {
    inner: Arc<RwLock<Inner>>,
}

#[derive(Debug, Default)]
struct Inner {
    /// Keyed by the display form of the edition id.
    editions: BTreeMap<String, Edition>,
    /// Keyed by interned encoding id; ordered by the id's string form.
    inclusions: BTreeMap<Id, EditionInclusion>,
}

impl EditionSession {
    /// Create a session variable with no declarations.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Declare an edition together with the encodings that join the family at it. Each
    /// added encoding's membership (`since`) is the declared edition; members of earlier
    /// editions are inherited and must not be restated.
    pub fn declare(&self, declaration: &EditionDeclaration) -> Result<(), EditionError> {
        self.declare_edition(declaration.edition)?;
        for encoding in declaration.added {
            self.declare_inclusion(EditionInclusion::new(*encoding, declaration.edition.id))?;
        }
        Ok(())
    }

    /// Declare an edition. Errors if an edition with the same id is already declared.
    pub fn declare_edition(&self, edition: Edition) -> Result<(), EditionError> {
        let mut inner = self.inner.write();
        let key = edition.id.to_string();
        if inner.editions.contains_key(&key) {
            return Err(EditionError::new(format!("duplicate edition {key}")));
        }
        inner.editions.insert(key, edition);
        Ok(())
    }

    /// Declare an edition inclusion. Errors if the encoding already has one: an encoding
    /// belongs to exactly one family, with one membership interval.
    pub fn declare_inclusion(&self, inclusion: EditionInclusion) -> Result<(), EditionError> {
        let mut inner = self.inner.write();
        if inner.inclusions.contains_key(&inclusion.encoding_id) {
            return Err(EditionError::new(format!(
                "duplicate edition inclusion for encoding {}",
                inclusion.encoding_id
            )));
        }
        inner.inclusions.insert(inclusion.encoding_id, inclusion);
        Ok(())
    }

    /// All declared editions, sorted by family and then chronologically. The newest frozen
    /// edition of each family is that family's `current` edition; unversioned editions are
    /// drafts.
    pub fn editions(&self) -> Vec<Edition> {
        let mut editions: Vec<Edition> = self.inner.read().editions.values().copied().collect();
        editions.sort_by_key(|e| (e.id.family, e.id.year, e.id.month, e.id.version));
        editions
    }

    /// Find a declared edition by id.
    pub fn find(&self, id: &EditionId) -> Option<Edition> {
        self.inner.read().editions.get(&id.to_string()).copied()
    }

    /// The newest frozen edition of a family, if any. Drafts are never current.
    pub fn current(&self, family: &str) -> Option<Edition> {
        self.editions()
            .into_iter()
            .filter(|e| e.id.family == family && !e.is_draft())
            .next_back()
    }

    /// Compute the full encoding set of an edition: every declared inclusion of the
    /// edition's family whose `since` is at or before it, sorted by encoding id.
    pub fn encodings_in(&self, edition: &EditionId) -> Vec<EditionInclusion> {
        // The map is keyed by encoding id, so the values are already sorted by it.
        self.inner
            .read()
            .inclusions
            .values()
            .filter(|inclusion| inclusion.since.is_at_or_before(edition))
            .copied()
            .collect()
    }

    /// Validate all registered declarations. Errors on inclusions referencing undeclared
    /// editions, editions out of chronological order within a family (unversioned drafts
    /// must be newest), malformed version strings, and members requiring a release newer
    /// than their edition declares.
    pub fn validate(&self) -> Result<(), EditionError> {
        let editions = self.editions();

        for edition in &editions {
            edition.id.validate()?;
            if let Some(version) = edition.min_vortex_version
                && parse_release(version).is_none()
            {
                return Err(EditionError::new(format!(
                    "edition {} declares malformed min_vortex_version {version:?}",
                    edition.id
                )));
            }
        }

        // Within each family, frozen editions must precede drafts: a frozen edition after
        // an unversioned one would imply the draft was skipped.
        for pair in editions.windows(2) {
            let (prev, next) = (&pair[0], &pair[1]);
            if prev.id.family == next.id.family && prev.is_draft() && !next.is_draft() {
                return Err(EditionError::new(format!(
                    "frozen edition {} follows draft {}; drafts must be newest in a family",
                    next.id, prev.id,
                )));
            }
        }

        let inner = self.inner.read();
        for inclusion in inner.inclusions.values() {
            inclusion.validate()?;

            let Some(edition) = inner.editions.get(&inclusion.since.to_string()) else {
                return Err(EditionError::new(format!(
                    "encoding {} is included in undeclared edition {}",
                    inclusion.encoding_id, inclusion.since
                )));
            };

            if let Some(required) = inclusion.required_vortex_release.and_then(parse_release)
                && let Some(declared) = edition.min_vortex_version.and_then(parse_release)
                && required > declared
            {
                return Err(EditionError::new(format!(
                    "encoding {} requires release {}, newer than edition {}'s declared \
                     min_vortex_version",
                    inclusion.encoding_id,
                    inclusion.required_vortex_release.unwrap_or_default(),
                    edition.id,
                )));
            }
        }

        Ok(())
    }
}

impl SessionVar for EditionSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Session data for Vortex editions.
pub trait EditionSessionExt: SessionExt {
    /// Returns the edition registry.
    fn editions(&self) -> SessionGuard<'_, EditionSession> {
        self.get::<EditionSession>()
    }
}

impl<S: SessionExt> EditionSessionExt for S {}
