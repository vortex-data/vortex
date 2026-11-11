// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Full-text, trigram and vector indexes for Vortex.
//!
//! Each index kind implements the `vortex.indexed` layout's
//! [`IndexVTable`](vortex_layout::layouts::indexed::IndexVTable), so the same kind can be stored in
//! two places:
//!
//! - **Inside a data file**, as a child of a `vortex.indexed` layout written by
//!   [`IndexedStrategy`](vortex_layout::layouts::indexed::IndexedStrategy). A scan's filters then
//!   consult the index automatically, and [`open_in_file_index`] opens it for ranked search.
//! - **As a standalone index file** next to the data, covering one or more data files, for a
//!   catalog to list and open with [`IndexFile`]. See [`standalone`].
//!
//! The kinds:
//!
//! | Kind | Answers | How |
//! |---|---|---|
//! | [`TrigramIndex`] | `LIKE` / `ILIKE` with literal runs of 3+ bytes | Prunes blocks lacking a trigram (superset) |
//! | [`InvertedIndex`] | [`text_match`] filters, BM25 top-k | Exact postings per token |
//! | [`IvfIndex`] | Cosine top-k over `f32` vectors | IVF lists with int8 codes (approximate) |
//!
//! Call [`initialize`] on a session before writing or reading files that use these indexes.

pub mod in_file;
pub mod inverted;
pub mod ivf;
mod keys;
mod postings;
pub mod rows;
pub mod standalone;
pub mod text_match;
pub mod tokenize;
pub mod trigram;

#[cfg(test)]
mod tests;

pub use in_file::open_in_file_index;
pub use inverted::InvertedIndex;
pub use ivf::IvfIndex;
pub use ivf::IvfOptions;
pub use rows::Hit;
pub use rows::IndexRows;
pub use rows::IndexStep;
pub use rows::Located;
pub use rows::StepFn;
pub use standalone::CoveredFile;
pub use standalone::FileHit;
pub use standalone::FileSelection;
pub use standalone::IndexFile;
pub use standalone::IndexFileBuilder;
pub use standalone::IndexFileInfo;
pub use text_match::TextMatch;
pub use text_match::text_match;
pub use trigram::TrigramIndex;
pub use trigram::TrigramOptions;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::indexed::INDEXED_LAYOUT_ID;
use vortex_layout::layouts::indexed::IndexSessionExt;
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

/// Register the index kinds, the `text_match` function, and the edition for writing
/// `vortex.indexed` layouts.
pub fn initialize(session: &VortexSession) -> VortexResult<()> {
    let indexes = session.indexes();
    indexes.register(TrigramIndex::new_ref());
    indexes.register(InvertedIndex::new_ref());
    indexes.register(IvfIndex::new_ref());
    session.scalar_fns().register(TextMatch);
    session
        .register_edition(&INDEX_EDITION_DECLARATION)
        .map_err(|err| vortex_err!("{err}"))?;
    session
        .enable_edition(INDEX_EDITION)
        .map_err(|err| vortex_err!("{err}"))?;
    Ok(())
}
