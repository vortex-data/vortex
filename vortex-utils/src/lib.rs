// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utility types and functions to be shared amongst the Vortex crates.

#![deny(missing_docs)]
#![allow(clippy::needless_pass_by_value)]

pub mod aliases;
pub mod debug_with;
#[cfg(feature = "dyn-traits")]
pub mod dyn_traits;
#[cfg(feature = "_test-harness")]
pub mod env;
pub mod iter;
