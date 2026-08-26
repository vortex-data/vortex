// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session variables for registered and enabled editions.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::ComponentKind;
use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionError;
use crate::EditionFamily;
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
    /// Keyed by family name.
    families: BTreeMap<String, EditionFamily>,
    /// Keyed by the display form of the edition id.
    editions: BTreeMap<String, Edition>,
    /// One map per member kind, each keyed by interned member id, because ids are only unique
    /// within a kind. An id may have one inclusion per family. Ordered by kind, then by the id's
    /// string form.
    inclusions: BTreeMap<ComponentKind, BTreeMap<Id, Vec<EditionInclusion>>>,
}

/// Registry of enabled editions, keyed by interned edition family.
type EditionsByFamily = ArcSwapMap<Id, EditionId>;

/// The editions enabled for writing in a session.
///
/// At most one edition is enabled per family. Enabling a newer or older edition from the
/// same family replaces the previous selection. This is separate from [`EditionSession`]:
/// registration describes what a session knows how to reason about, while enabling is the
/// explicit writer policy.
///
/// Backed by an [`ArcSwapMap`] keyed by edition family, so clones observe the same selection
/// and enabling an edition replaces the family's previous entry.
#[derive(Clone, Debug, Default)]
pub struct EnabledEditions {
    inner: EditionsByFamily,
}

impl EnabledEditions {
    /// Return the enabled editions.
    pub fn editions(&self) -> Vec<EditionId> {
        self.inner.read(|map| map.values().copied().collect())
    }

    fn enable(&self, edition: EditionId) {
        // The family is a `&'static str`; `Into<Id>` interns it once at enable time (a rare
        // config-time write, never on the read path).
        self.inner.insert(Id::from(edition.family), edition);
    }
}

impl EditionSession {
    /// Create a session variable with no declarations.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Declare an edition together with the members added at it. Each entry's membership
    /// (`since`) is the declared edition; earlier entries are inherited and must not be restated.
    pub fn declare(&self, declaration: &EditionDeclaration) -> Result<(), EditionError> {
        self.declare_edition(declaration.edition)?;
        for member in declaration.added {
            self.declare_inclusion(EditionInclusion::new(
                member.kind,
                member.component,
                declaration.edition.id,
            ))?;
        }
        Ok(())
    }

    /// Declare an edition family. Errors if a family with the same name is already
    /// declared. Every family must be declared before [`EditionSession::validate`] will
    /// accept editions belonging to it.
    pub fn declare_family(&self, family: &EditionFamily) -> Result<(), EditionError> {
        let mut inner = self.inner.write();
        if inner.families.contains_key(family.name) {
            return Err(EditionError::new(format!(
                "duplicate edition family {}",
                family.name
            )));
        }
        inner.families.insert(family.name.to_string(), *family);
        Ok(())
    }

    /// All declared families, sorted by name.
    pub fn families(&self) -> Vec<EditionFamily> {
        self.inner.read().families.values().copied().collect()
    }

    /// Find a declared family by name.
    pub fn find_family(&self, name: &str) -> Option<EditionFamily> {
        self.inner.read().families.get(name).copied()
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

    /// Declare an edition inclusion. A component may belong to multiple families but joins each
    /// family only once. A newer wire representation uses a new component ID. Kind is part of the
    /// key, so an array encoding and a layout may share an id.
    pub fn declare_inclusion(&self, inclusion: EditionInclusion) -> Result<(), EditionError> {
        let mut inner = self.inner.write();
        let by_id = inner.inclusions.entry(inclusion.kind).or_default();
        let history = by_id.entry(inclusion.component_id).or_default();
        let previous = history
            .iter()
            .filter(|existing| existing.since.family == inclusion.since.family)
            .max_by_key(|existing| {
                (
                    existing.since.year,
                    existing.since.month,
                    existing.since.version,
                )
            });

        if let Some(previous) = previous {
            return Err(EditionError::new(format!(
                "{} {} already joined family {} in edition {}",
                inclusion.kind, inclusion.component_id, inclusion.since.family, previous.since,
            )));
        }
        history.push(inclusion);
        history.sort_by_key(|entry| {
            (
                entry.since.family,
                entry.since.year,
                entry.since.month,
                entry.since.version,
            )
        });
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
            .rfind(|e| e.id.family == family && !e.is_draft())
    }

    /// Compute an edition's members of one kind, sorted by component id. For each id, this returns
    /// its inclusion in the edition's family when it joined at or before the requested edition.
    /// Only that kind's declarations are scanned.
    pub fn components_in(&self, edition: &EditionId, kind: ComponentKind) -> Vec<EditionInclusion> {
        let inner = self.inner.read();
        let Some(by_id) = inner.inclusions.get(&kind) else {
            return vec![];
        };
        by_id
            .values()
            .filter_map(|history| {
                history
                    .iter()
                    .filter(|inclusion| inclusion.since.is_at_or_before(edition))
                    .max_by_key(|inclusion| {
                        (
                            inclusion.since.year,
                            inclusion.since.month,
                            inclusion.since.version,
                        )
                    })
                    .copied()
            })
            .collect()
    }

    /// Validate all registered declarations. Errors on editions in undeclared families,
    /// inclusions referencing undeclared editions, editions out of chronological order within
    /// a family (unversioned drafts must be newest), malformed version strings, and members
    /// requiring a release newer than their edition declares.
    pub fn validate(&self) -> Result<(), EditionError> {
        let editions = self.editions();

        for family in self.families() {
            family.validate()?;
        }

        for edition in &editions {
            edition.id.validate()?;
            if self.find_family(edition.id.family).is_none() {
                return Err(EditionError::new(format!(
                    "edition {} belongs to undeclared family {}; declare the family before \
                     its editions",
                    edition.id, edition.id.family,
                )));
            }
            if let Some(version) = edition.min_library_version
                && parse_release(version).is_none()
            {
                return Err(EditionError::new(format!(
                    "edition {} declares malformed min_library_version {version:?}",
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
        for inclusion in inner
            .inclusions
            .values()
            .flat_map(|by_id| by_id.values())
            .flatten()
        {
            inclusion.validate()?;

            let Some(edition) = inner.editions.get(&inclusion.since.to_string()) else {
                return Err(EditionError::new(format!(
                    "{} {} is included in undeclared edition {}",
                    inclusion.kind, inclusion.component_id, inclusion.since
                )));
            };

            if let Some(required) = inclusion.required_vortex_release.and_then(parse_release)
                && let Some(declared) = edition.min_library_version.and_then(parse_release)
                && required > declared
            {
                return Err(EditionError::new(format!(
                    "{} {} requires release {}, newer than edition {}'s declared \
                     min_library_version",
                    inclusion.kind,
                    inclusion.component_id,
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

impl SessionVar for EnabledEditions {
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

    /// Returns the editions enabled for writing.
    ///
    /// Accessing this method installs the enabled-editions session variable if it is absent, with
    /// an initially empty selection.
    fn enabled_editions(&self) -> SessionGuard<'_, EnabledEditions> {
        self.get::<EnabledEditions>()
    }

    /// Register an edition declaration with this session.
    fn register_edition(&self, declaration: &EditionDeclaration) -> Result<(), EditionError> {
        self.editions().declare(declaration)
    }

    /// Enable a registered edition for writing.
    ///
    /// Enabling an edition replaces the enabled edition from the same family. An edition
    /// must be registered first so a typo or unavailable third-party declaration cannot
    /// silently produce an empty writable set.
    fn enable_edition(&self, edition: EditionId) -> Result<(), EditionError> {
        if self.editions().find(&edition).is_none() {
            return Err(EditionError::new(format!(
                "cannot enable unregistered edition {edition}"
            )));
        }
        self.enabled_editions().enable(edition);
        Ok(())
    }

    /// Resolve the ids of one [`ComponentKind`] across all enabled editions: what a writer may
    /// emit for that kind.
    ///
    /// Ids are only unique within a kind, so this never mixes kinds. An empty result means the
    /// enabled editions permit no components of this kind.
    fn enabled_component_ids(&self, kind: ComponentKind) -> Vec<Id> {
        let Some(enabled) = self.get_opt::<EnabledEditions>() else {
            return vec![];
        };
        let editions = self.editions();
        let mut ids: Vec<Id> = enabled
            .editions()
            .iter()
            .flat_map(|edition| editions.components_in(edition, kind))
            .map(|inclusion| inclusion.component_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

impl<S: SessionExt> EditionSessionExt for S {}
