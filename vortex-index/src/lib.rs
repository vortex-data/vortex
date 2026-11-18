// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! **Experimental proof of concept.** This crate is not a supported index library. It exists to
//! show how indexes can be built on Vortex's scanning API, and its API may change or be removed
//! without notice. It is not published, and no other crate in the workspace depends on it.
//!
//! The indexes it will hold are ordinary Vortex files: an index is written with the Vortex
//! writer and read back through scan v2, and a search is a sequence of scans over the index file
//! followed by scans of only the data rows it selects. Run as planners under a plan driver, the
//! index reads and the data reads they lead to are scheduled and coalesced like any other scan.
//!
//! Planned contents, each added separately:
//!
//! - the shared machinery: index rows read through scan v2, searches expressed as steps of
//!   reads, and standalone index files bound to the data files they cover;
//! - index kinds: a trigram index for `LIKE`, an inverted index for full-text search and BM25,
//!   and an IVF index for vector search;
//! - a planner that turns an index search into a plan over the data files it selects, and an
//!   example that runs one.
