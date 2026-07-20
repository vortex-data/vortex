// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions of Vortex *editions*: named, frozen sets of encodings that a writer may put in
//! a file, carrying a forever read-compatibility guarantee.
//!
//! This crate is the single source of truth for editions. It holds two pieces of data:
//!
//! 1. [`EDITIONS`] — the edition list itself: identifiers and draft flags. The freeze date
//!    is carried by the identifier (`core2026.07.0` was frozen in 2026-07).
//! 2. [`ENCODINGS`] — the per-encoding membership declarations: which edition each encoding
//!    has been a member of *since*, and the release required to read it.
//!
//! Everything else is computed: [`manifest::compute_manifests`] derives the full,
//! closure-validated encoding set for every edition, and [`generate`] turns those manifests
//! into machine-readable JSON files and the published documentation pages (via
//! `cargo xtask generate-editions`).
//!
//! The crate deliberately depends on nothing beyond `serde`, so it can be consumed by the
//! writer, `xtask`, the compat-test suite, and language bindings alike. Encoding IDs are plain
//! strings; tests elsewhere in the workspace assert that every declared ID resolves to a
//! registered encoding in the default session.
//!
//! See the published spec at <https://docs.vortex.dev/specs/editions.html> and the internal
//! design notes in `docs/developer-guide/internals/editions.md`.

mod definitions;
pub mod generate;
pub mod manifest;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

pub use definitions::CORE_2026_07_0;
pub use definitions::CORE_2026_10_0;
pub use definitions::EDITIONS;
pub use definitions::ENCODINGS;

/// The identifier of an edition, e.g. `core2026.07.0`.
///
/// The `family` names an independently versioned, additive group of encodings (`core` is the
/// set the default writer emits). The date components record when the edition was frozen and
/// order editions chronologically *within* a family; there is no ordering across families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Create an edition identifier.
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

/// An edition: a named, frozen set of encodings with a read-compatibility guarantee.
///
/// The encoding sets themselves are not stored here — they are computed from the
/// per-encoding [`EncodingDecl::since`] declarations by [`manifest::compute_manifests`].
#[derive(Clone, Copy, Debug)]
pub struct Edition {
    /// The edition identifier. Also carries the freeze date: `core2026.07.0` was frozen in
    /// 2026-07.
    pub id: EditionId,
    /// Drafts are editions being assembled: they carry no guarantee, may change freely, and
    /// are never the default write target.
    pub draft: bool,
}

/// Declares an encoding's membership of an edition family.
///
/// An encoding declared with `since: E` is a member of `E` and of every later edition of the
/// same family (until deprecation exists). These declarations currently live centrally in
/// this crate; they are expected to migrate next to each encoding's vtable once the plugin
/// trait grows edition metadata.
#[derive(Clone, Copy, Debug)]
pub struct EncodingDecl {
    /// The encoding ID, e.g. `vortex.alp`. Globally unique across everything an edition can
    /// cover: when layout encodings join editions, their IDs must be distinct from array
    /// encoding IDs.
    pub id: &'static str,
    /// The first edition this encoding is a member of.
    pub since: EditionId,
    /// The earliest Vortex release able to read and execute this encoding, recorded on the
    /// membership edge from evidence (e.g. compat-fixture history). `None` until recorded;
    /// an edition's required release is derived as the maximum over its members' recorded
    /// releases, falling back to the first release containing the edition when any edge is
    /// unrecorded.
    pub required_vortex_release: Option<&'static str>,
}

/// Error raised when edition definitions are inconsistent or generation fails.
#[derive(Debug)]
pub struct EditionError(String);

impl EditionError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl Display for EditionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for EditionError {}
