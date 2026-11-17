// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** This crate is not a supported index library. It exists to
//! show how indexes can be built on Vortex's scanning API, and its API may change or be removed
//! without notice. It is not published, and no other crate in the workspace depends on it.
//!
//! An index is an ordinary Vortex file. It is written with the Vortex writer from the
//! batches of an index kind, which implements the `vortex.indexed` layout's
//! [`IndexVTable`](vortex_layout::layouts::indexed::IndexVTable), and read back through scan v2.
//! A lookup takes a filter over the indexed column and answers with the rows of each covered data
//! file that may match, ready to pass to that file's scan. The crate plans scans and never runs
//! them: the caller's driver does. See [`standalone`].

pub mod standalone;

#[cfg(test)]
mod tests;

pub use standalone::CoveredFile;
pub use standalone::FileMatch;
pub use standalone::FileRows;
pub use standalone::IndexFile;
pub use standalone::IndexFileBuilder;
pub use standalone::Lookup;
