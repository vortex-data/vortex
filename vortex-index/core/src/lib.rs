// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** This crate is not a supported index library. It exists to
//! show how indexes can be built on Vortex's scanning API, and its API may change or be removed
//! without notice. It is not published, and no crate outside `vortex-index/` depends on it.
//!
//! The shared machinery of the index kinds, which live in their own crates:
//!
//! | Crate | Kind | Answers |
//! |---|---|---|
//! | `vortex-index-trigram` | `TrigramIndex` | `LIKE` / `ILIKE` with literal runs of 3+ bytes, as a superset |
//! | `vortex-index-inverted` | `InvertedIndex` | `text_match` filters exactly, BM25 top-k |
//! | `vortex-index-ivf` | `IvfIndex` | Cosine top-k over `f32` vectors (approximate) |
//!
//! Each kind implements the `vortex.indexed` layout's
//! [`IndexVTable`](vortex_layout::layouts::indexed::IndexVTable), and [`IndexKind`] on top of it,
//! so the same kind can be stored in two places:
//!
//! - **Inside a data file**, as a child of a `vortex.indexed` layout written by
//!   [`IndexedStrategy`](vortex_layout::layouts::indexed::IndexedStrategy). A scan's filters then
//!   consult the index, and [`open_in_file_index`] opens it for ranked search.
//! - **As a standalone index file** next to the data, covering one or more data files, for a
//!   catalog to list and open with [`IndexFile`]. See [`standalone`].
//!
//! Either way an index is read through scan v2, as [`IndexRows`], and a search is a chain of
//! [`IndexStep`]s: scans of index rows to run, then the answer.
//!
//! Call [`initialize`] on a session, and register the kinds from their crates, before writing or
//! reading files that use these indexes.

pub mod in_file;
pub mod keys;
pub mod plan;
pub mod postings;
pub mod rows;
pub mod standalone;
pub mod tokenize;

pub use in_file::open_in_file_index;
pub use rows::Hit;
pub use rows::IndexKind;
pub use rows::IndexRows;
pub use rows::IndexStep;
pub use rows::Located;
pub use rows::QueryKeys;
pub use rows::StepFn;
pub use standalone::CoveredFile;
pub use standalone::FileHit;
pub use standalone::FileSelection;
pub use standalone::IndexFile;
pub use standalone::IndexFileBuilder;
pub use standalone::IndexFileInfo;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::INDEXED_LAYOUT_ID;
use vortex_session::VortexSession;

/// The edition that lets a file writer emit the `vortex.indexed` layout.
pub const INDEX_EDITION: EditionId = EditionId::new("vortex-index", 2026, 1, 0);

static INDEX_EDITION_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: INDEX_EDITION,
        immutable: false,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&INDEXED_LAYOUT_ID)],
};

/// Register and enable the edition for writing `vortex.indexed` layouts.
///
/// The index kinds are registered by their own crates.
pub fn initialize(session: &VortexSession) -> VortexResult<()> {
    session
        .register_edition(&INDEX_EDITION_DECLARATION)
        .map_err(|err| vortex_err!("{err}"))?;
    session
        .enable_edition(INDEX_EDITION)
        .map_err(|err| vortex_err!("{err}"))?;
    Ok(())
}
