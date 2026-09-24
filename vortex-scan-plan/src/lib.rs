// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prototype of the scan planner protocol described in `vortex-scan/design/PROTOTYPE.md`.
//!
//! The crate holds the protocol traits (`planner`, `morsel`, `io`, `next`), a single-worker
//! blocking [`driver`], and the file [`stages`] that compose into `plan_file`.

pub mod driver;
pub mod io;
pub mod morsel;
pub mod next;
pub mod planner;
pub mod stages;

#[cfg(test)]
mod tests;
