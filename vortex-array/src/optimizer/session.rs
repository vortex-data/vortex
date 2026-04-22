// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for pluggable parent-reduce rules.
//!
//! [`OptimizerSession`] wraps an [`FnRegistry`] keyed by `(parent_encoding_id, child_encoding_id)`
//! and is consulted by the optimizer during execution, before the child encoding's static
//! `PARENT_RULES` are tried. Entries are typed as [`ReduceParentFn`](super::ReduceParentFn).
//!
//! The registry is empty by default. Downstream crates register `ReduceParentFn` values to add
//! new parent-reduce rules or override ones that the child encoding would otherwise run from its
//! static `PARENT_RULES`.

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::FnRegistry;

/// Session state for pluggable parent-reduce dispatch keyed by `(parent_id, child_id)`.
#[derive(Debug, Default)]
pub struct OptimizerSession {
    registry: FnRegistry,
}

impl OptimizerSession {
    /// Create an empty session with no rules registered.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Access the underlying registry for direct registration and lookup.
    pub fn registry(&self) -> &FnRegistry {
        &self.registry
    }
}

/// Extension trait for accessing the optimizer registry from a Vortex session.
pub trait OptimizerSessionExt: SessionExt {
    /// Returns the optimizer session variable.
    fn optimizer(&self) -> Ref<'_, OptimizerSession> {
        self.get::<OptimizerSession>()
    }
}
impl<S: SessionExt> OptimizerSessionExt for S {}
