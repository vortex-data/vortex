// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FileCheck-style checks of the array encodings in Vortex files.
//!
//! A Vortex file is flattened into *units*, one per flat layout (the leaves of the layout tree
//! that hold a serialized array). Each unit is rendered as text: a header line, then the array
//! encoding tree.
//!
//! ```text
//! unit: l_comment/data[3] rows=8192 layouts=vortex.struct/vortex.zoned/vortex.chunked/vortex.flat
//! root: vortex.fsst(utf8, len=8192)
//!   uncompressed_lengths: fastlanes.bitpacked.v2(u8, len=8192)
//!   codes_offsets: fastlanes.bitpacked.v2(u32, len=8193)
//! ```
//!
//! The unit path joins the layout child names: struct fields with `.`, chunks with `[i]`, and
//! other children (`data`, `zones`, `values`, `codes`, ...) with `/`. Rendering decodes each
//! serialized array but decompresses nothing.
//!
//! # Check files
//!
//! A check file holds [filecheck](https://docs.rs/filecheck) directives, which match the
//! rendered text partially and in order, like LLVM FileCheck:
//!
//! - `check: <pattern>` matches after the previous match; `nextln:` and `sameln:` match on the
//!   next line or the same line; `unordered:` matches in any order.
//! - `not: <pattern>` requires the pattern to be absent between the surrounding matches, or
//!   after the last one.
//! - `regex: VAR=<regex>` defines a variable that patterns use as `$VAR`. `$(VAR=<regex>)`
//!   defines one inline, and `$$` is a literal `$`.
//!
//! Patterns are literal text otherwise. A pattern that starts or ends with a letter, digit or
//! `_` only matches at a word boundary there: `vortex.decimal` does not match inside
//! `vortex.decimal_byte_parts`, but `fastlanes.bitpacked` matches `fastlanes.bitpacked.v2`.
//!
//! The check file adds two directives of its own:
//!
//! - `files: <regex>` names the Vortex files the check file applies to. Each regex must match a
//!   whole path, relative to the root directory the caller searches (see [`check_tree`]).
//!   `files:` lines must come before the first `units:` line.
//! - `units: <glob>` starts a section whose directives are checked against *every* unit whose
//!   path matches the glob, separately. A section that matches no unit fails, so a typo in a
//!   glob cannot pass silently. Directives before the first `units:` line are checked once
//!   against the dump of the whole file.
//!
//! In globs, `*` matches any run of characters, including `/`, `.` and `[`, and `?` matches one
//! character. Everything else, including `[` and `]`, matches literally. So `l_comment/data*`
//! matches every chunk of a chunked column (`l_comment/data[0]`, ...) and the single unit of
//! an unchunked one (`l_comment/data`).
//!
//! Every line must be blank, a `#` comment, or one of these directives at the start of the
//! line, so a misspelled directive is an error rather than a check that silently never runs.
//!
//! ```text
//! files: tpch/data/lineitem\.vortex
//!
//! # Every chunk of l_comment must be FSST-compressed.
//! units: l_comment/data*
//! check: root: vortex.fsst
//!
//! # Dates must never be stored as uncompressed primitives.
//! units: l_shipdate/data*
//! not: vortex.primitive
//! ```
//!
//! # Entry points
//!
//! - [`check_tree`] checks every Vortex file under a directory that a check file's `files:`
//!   regexes match. It is the entry point for test suites, benchmark runners and data
//!   generators.
//! - [`load_units`] and [`run_checks`] check one file, as `vx filecheck` does.

mod check;
#[cfg(not(target_arch = "wasm32"))]
mod tree;
mod units;

pub use check::CheckFile;
pub use check::CheckSection;
pub use check::Directive;
pub use check::Failure;
pub use check::Glob;
pub use check::SectionOutcome;
pub use check::format_report;
pub use check::parse_check_file;
pub use check::read_check_file;
pub use check::run_checks;
#[cfg(not(target_arch = "wasm32"))]
pub use tree::FileOutcome;
#[cfg(not(target_arch = "wasm32"))]
pub use tree::check_tree;
#[cfg(not(target_arch = "wasm32"))]
pub use tree::format_tree_report;
#[cfg(not(target_arch = "wasm32"))]
pub use tree::matching_files;
pub use units::DumpOptions;
pub use units::Unit;
#[cfg(not(target_arch = "wasm32"))]
pub use units::check_vortex_file;
pub use units::collect_units;
#[cfg(not(target_arch = "wasm32"))]
pub use units::load_units;

#[cfg(test)]
mod tests;
