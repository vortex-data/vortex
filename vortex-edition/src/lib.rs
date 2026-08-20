// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions of Vortex *editions*: named, frozen sets of components that a writer may put
//! in a file, carrying a forever read-compatibility guarantee.
//!
//! Editions live on the session, like encodings do: [`EditionSession`] holds the registered
//! editions and [`EnabledEditions`] selects which of them a writer may emit. Declarations
//! are plain constants — an [`EditionId`] plus an [`Edition`] record, and one
//! [`EditionInclusion`] per member stating that it is a member of an edition *and every
//! later edition of the same family*. Any crate can register declarations into a session,
//! so inclusions can live next to the component they describe.
//!
//! Every membership is typed by a [`ComponentKind`], and members are resolved one kind at a
//! time with [`EditionSessionExt::enabled_component_ids`]: the file writer restricts the
//! arrays, layouts, extension dtypes, and aggregates it writes from separate id sets, never one
//! untyped set.
//!
//! An edition is a **draft** until its [`Edition::min_vortex_version`] is recorded —
//! recording it is the act of freezing. The per-edition member sets are computed from the
//! registered declarations by [`EditionSession::components_in`], and correctness is enforced
//! by unit tests: [`EditionSession::validate`] checks a whole registry, and
//! [`test_harness::validate_edition`] validates one edition's constraints — call it once in
//! the `#[cfg(test)]` module of each edition definition.
//!
//! The first-party edition declarations live in the public `vortex` crate, which registers
//! and enables them on the default session. See the published spec at
//! <https://docs.vortex.dev/specs/editions.html>.

mod session;
pub mod test_harness;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

pub use session::EditionSession;
pub use session::EditionSessionExt;
pub use session::EnabledEditions;
use vortex_session::registry::Id;

/// The identifier of an edition, e.g. `core2026.07.0`.
///
/// The `family` names an independently versioned, additive group of components (`core` is the
/// set the default writer emits). The date components record when the edition was frozen and
/// order editions chronologically *within* a family; there is no ordering across families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EditionId {
    /// The edition family, e.g. `core`.
    pub family: &'static str,
    /// Year the edition was cut.
    pub year: u16,
    /// Month the edition was cut.
    pub month: u8,
    /// Distinguishes editions cut in the same month; normally `0`.
    pub version: u8,
}

impl EditionId {
    /// Create an edition identifier. Validated by [`EditionId::validate`], which
    /// [`test_harness::validate_edition`] exercises
    /// per edition in unit tests.
    pub const fn new(family: &'static str, year: u16, month: u8, version: u8) -> Self {
        Self {
            family,
            year,
            month,
            version,
        }
    }

    /// Returns true if `self` is the same edition as `other` or an earlier edition of the
    /// same family. Editions of different families are never ordered.
    pub fn is_at_or_before(&self, other: &EditionId) -> bool {
        self.family == other.family
            && (self.year, self.month, self.version) <= (other.year, other.month, other.version)
    }

    /// Validate the identifier's form: a non-empty lowercase family, a four-digit year,
    /// and a month in 01-12. Checked for every declared edition by
    /// [`EditionSession::validate`] and per edition by
    /// [`test_harness::validate_edition`].
    pub fn validate(&self) -> Result<(), EditionError> {
        if self.family.is_empty() || !self.family.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(EditionError::new(format!(
                "edition {self} must have a non-empty lowercase family, e.g. `core`"
            )));
        }
        if !(1000..=9999).contains(&self.year) {
            return Err(EditionError::new(format!(
                "edition {self} must have a four-digit year"
            )));
        }
        if !(1..=12).contains(&self.month) {
            return Err(EditionError::new(format!(
                "edition {self} must have a month in 01-12"
            )));
        }
        Ok(())
    }
}

impl Display for EditionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}.{:02}.{}",
            self.family, self.year, self.month, self.version
        )
    }
}

/// The kind of component an edition membership covers.
///
/// Ids are unique per kind, not globally: a layout named `vortex.flat` and an array named
/// `vortex.flat` are different members. Every membership records its kind, and the writer
/// resolves one kind at a time, so the set restricting written arrays never restricts
/// written layouts. Further kinds (scalar functions, say) can be added the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentKind {
    /// An array encoding, e.g. `vortex.alp`, registered in the session's array registry.
    Array,
    /// A layout encoding, e.g. `vortex.flat`, registered in the session's layout registry.
    Layout,
    /// An extension dtype, e.g. `vortex.timestamp`, registered in the session's dtype registry.
    DType,
    /// An aggregate function, e.g. `vortex.min`, written into zone maps and registered in
    /// the session's aggregate function registry.
    Aggregate,
}

impl Display for ComponentKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Array => "array",
            Self::Layout => "layout",
            Self::DType => "dtype",
            Self::Aggregate => "aggregate",
        })
    }
}

/// An edition: a named set of components with a read-compatibility guarantee, registered with
/// [`EditionSession::declare_edition`]. The set itself is computed from the registered
/// [`EditionInclusion`]s by [`EditionSession::components_in`].
#[derive(Clone, Copy, Debug)]
pub struct Edition {
    /// The edition identifier. Also carries the freeze date: `core2026.07.0` freezes in
    /// 2026-07.
    pub id: EditionId,
    /// The minimum Vortex version whose reader supports every member of this edition.
    ///
    /// Recording this is the act of freezing: an edition with `None` is a **draft** — being
    /// assembled, carrying no guarantee, free to change, never the default write target.
    /// Validated against the members' [`EditionInclusion::required_vortex_release`] values:
    /// no member may require a version newer than the edition declares.
    pub min_vortex_version: Option<&'static str>,
}

impl Edition {
    /// A draft is an edition whose `min_vortex_version` has not been recorded yet.
    pub fn is_draft(&self) -> bool {
        self.min_vortex_version.is_none()
    }
}

/// Declares that a component is a member of an edition — and of every later edition of the
/// same family. Registered with [`EditionSession::declare_inclusion`].
#[derive(Clone, Copy, Debug)]
pub struct EditionInclusion {
    /// What the membership covers. Ids are unique per kind, so this is part of the
    /// member's identity, not a label.
    pub kind: ComponentKind,
    /// The interned component id, e.g. `vortex.alp`.
    pub component_id: Id,
    /// The first edition this component is a member of.
    pub since: EditionId,
    /// The earliest Vortex release able to read and execute this component, recorded from
    /// evidence (e.g. compat-fixture history). `None` until recorded.
    pub required_vortex_release: Option<&'static str>,
}

/// A source of a component id for edition declarations.
///
/// Implemented for raw id strings (`"vortex.alp"`) and interned [`Id`]s here; encoding
/// vtables implement it where they are defined, so a declaration can name the vtable
/// (`&Primitive`) instead of spelling its id. The id alone does not say what kind of
/// component it names — [`EditionMember`] pairs it with a [`ComponentKind`].
pub trait AsComponentId: Debug + Send + Sync {
    /// The interned component id.
    fn component_id(&self) -> Id;
}

impl AsComponentId for str {
    #[expect(
        clippy::disallowed_methods,
        reason = "interning a dynamic component id at declaration time"
    )]
    fn component_id(&self) -> Id {
        Id::new(self)
    }
}

impl AsComponentId for Id {
    fn component_id(&self) -> Id {
        *self
    }
}

// `str` is unsized and cannot be a trait object, so declaration blocks name components as
// `&"vortex.alp"` through this impl.
impl AsComponentId for &'static str {
    fn component_id(&self) -> Id {
        (**self).component_id()
    }
}

/// A component that joins an edition, named by id string or vtable and tagged with the kind
/// of registry it belongs to. Built with the per-kind constructors, so a declaration reads
/// as `EditionMember::array(&"vortex.alp")`.
#[derive(Clone, Copy, Debug)]
pub struct EditionMember {
    /// What kind of component this is.
    pub kind: ComponentKind,
    /// The component, named by id string or by vtable.
    pub component: &'static dyn AsComponentId,
}

impl EditionMember {
    /// An array encoding member, e.g. `vortex.alp`.
    pub const fn array(component: &'static dyn AsComponentId) -> Self {
        Self {
            kind: ComponentKind::Array,
            component,
        }
    }

    /// A layout member, e.g. `vortex.flat`.
    pub const fn layout(component: &'static dyn AsComponentId) -> Self {
        Self {
            kind: ComponentKind::Layout,
            component,
        }
    }

    /// An extension dtype member, e.g. `vortex.timestamp`.
    pub const fn dtype(component: &'static dyn AsComponentId) -> Self {
        Self {
            kind: ComponentKind::DType,
            component,
        }
    }

    /// An aggregate function member, e.g. `vortex.min`.
    pub const fn aggregate(component: &'static dyn AsComponentId) -> Self {
        Self {
            kind: ComponentKind::Aggregate,
            component,
        }
    }
}

/// Declares an edition together with the components that join the family at it, in one
/// block. Registered with [`EditionSession::declare`], which derives each member's
/// membership (`since` = the declared edition) from the block structure.
#[derive(Clone, Copy, Debug)]
pub struct EditionDeclaration {
    /// The edition being declared.
    pub edition: Edition,
    /// The components that join the family at this edition, each tagged with its
    /// [`ComponentKind`]. Members of earlier editions are inherited and never restated.
    pub added: &'static [EditionMember],
}

impl EditionInclusion {
    /// Declare that a component of `kind` is a member of `since` and every later edition of
    /// the same family. The component can be named by id string or by vtable.
    pub fn new<C: AsComponentId + ?Sized>(
        kind: ComponentKind,
        component: &C,
        since: EditionId,
    ) -> Self {
        Self {
            kind,
            component_id: component.component_id(),
            since,
            required_vortex_release: None,
        }
    }

    /// Declare that an array encoding is a member of `since` and every later edition of the
    /// same family.
    pub fn array<C: AsComponentId + ?Sized>(encoding: &C, since: EditionId) -> Self {
        Self::new(ComponentKind::Array, encoding, since)
    }

    /// Declare that an extension dtype is a member of `since` and every later edition of the
    /// same family.
    pub fn dtype<C: AsComponentId + ?Sized>(dtype: &C, since: EditionId) -> Self {
        Self::new(ComponentKind::DType, dtype, since)
    }

    /// Validate the declaration's form: a lowercase `namespace.name` component id and, if
    /// recorded, a well-formed `major.minor.patch` release. Checked for every declared
    /// inclusion by [`EditionSession::validate`].
    pub fn validate(&self) -> Result<(), EditionError> {
        let id = self.component_id.as_str();
        let well_formed = !id.starts_with('.')
            && !id.ends_with('.')
            && id.contains('.')
            && id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c));
        if !well_formed {
            return Err(EditionError::new(format!(
                "invalid {} id {id:?}: expected lowercase `namespace.name`, e.g. `vortex.alp`",
                self.kind
            )));
        }
        if let Some(release) = self.required_vortex_release
            && parse_release(release).is_none()
        {
            return Err(EditionError::new(format!(
                "{} {id} declares malformed required_vortex_release {release:?}",
                self.kind
            )));
        }
        Ok(())
    }
}

/// Parse a `major.minor.patch` release string into a comparable key.
pub(crate) fn parse_release(release: &str) -> Option<Vec<u64>> {
    let parts: Vec<u64> = release
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<_>>()?;
    (parts.len() == 3).then_some(parts)
}

/// Error raised when edition declarations are inconsistent.
#[derive(Debug)]
pub struct EditionError(String);

impl EditionError {
    /// Create an error with the given message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl Display for EditionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for EditionError {}
