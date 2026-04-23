// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session state for pluggable parent-reduce rules.
//!
//! [`ArrayKernels`] wraps an [`FnRegistry`] keyed by `(parent_encoding_id, child_encoding_id)`
//! and is consulted by the optimizer during execution, before the child encoding's static
//! `PARENT_RULES` are tried. Entries are typed as [`ReduceParentFn`](super::ReduceParentFn).
//!
//! The registry is empty by default. Downstream crates register `ReduceParentFn` values to add
//! new parent-reduce rules or override ones that the child encoding would otherwise run from its
//! static `PARENT_RULES`.

use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::FnRegistry;
use vortex_session::registry::Id;
use vortex_utils::aliases::DefaultHashBuilder;

use crate::ArrayRef;

static FN_HASHER: LazyLock<DefaultHashBuilder> = LazyLock::new(DefaultHashBuilder::default);

/// Pluggable parent-reduce function signature used by [`ArrayKernels`].
///
/// A function of this type rewrites the parent array that holds `child` at `child_idx`, given
/// the child itself and its parent. Returns `Ok(None)` when the function doesn't apply.
///
/// Registered under `(parent_encoding_id, child_encoding_id)`; callers downcast the erased
/// `child`/`parent` to their expected types before applying logic.
pub type ReduceParentFn =
    fn(child: &ArrayRef, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FnKind {
    Reduce,
    ReduceParent,
    ExecuteParent,
    Execute,
}

/// Session state for pluggable parent-reduce dispatch keyed by `(parent_id, child_id)`.
#[derive(Debug, Default)]
pub struct ArrayKernels {
    registry: FnRegistry,
}

impl ArrayKernels {
    /// Create an empty session with no rules registered.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn register_reduce_parent(&self, outer: Id, child: Id, f: ReduceParentFn) {
        self.registry
            .register(self.hash_fn_ids(outer, child, FnKind::ReduceParent), f)
    }

    pub fn find_reduce_parent(&self, outer: Id, child: Id) -> Option<Arc<ReduceParentFn>> {
        self.registry
            .find(self.hash_fn_ids(outer, child, FnKind::ReduceParent))
    }

    pub fn contains_reduce_parent(&self, outer: Id, child: Id) -> bool {
        self.registry
            .contains(self.hash_fn_ids(outer, child, FnKind::ReduceParent))
    }

    fn hash_fn_ids(&self, outer: Id, child: Id, fn_kind: FnKind) -> u64 {
        FN_HASHER.hash_one((outer, child, fn_kind))
    }
}

/// Extension trait for accessing the optimizer registry from a Vortex session.
pub trait ArrayKernelsExt: SessionExt {
    /// Returns the optimizer session variable.
    fn kernels(&self) -> Ref<'_, ArrayKernels> {
        self.get::<ArrayKernels>()
    }
}

impl<S: SessionExt> ArrayKernelsExt for S {}
