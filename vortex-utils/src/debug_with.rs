// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities for implementing `Debug` via a closure.
//!
//! This provides a stable alternative to the `debug_closures` feature: <https://github.com/rust-lang/rust/issues/117729>.

use std::fmt;

/// A wrapper that implements `Debug` via a closure.
pub struct DebugWith<F>(pub F)
where
    F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result;

impl<F> fmt::Debug for DebugWith<F>
where
    F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0)(f)
    }
}
