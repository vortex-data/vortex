// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in extension dtypes with logical semantics beyond their storage.
//!
//! Temporal dtypes attach units to primitive storage. Wide integers give fixed-size byte lists
//! signed numeric ordering, so their operations must respect the extension dtype.

pub mod datetime;

pub mod integer;

#[cfg(test)]
mod tests;
