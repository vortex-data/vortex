// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The integration tests of `vortex-arrow`, linked into one test executable.
//!
//! Add a new integration test as a module here, not as a new file in `tests/`. Each file in
//! `tests/` is a separate executable that links the full dependency graph of the crate.

mod canonical;
mod vortex_id;
