// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of pluggable array kernels.
//!
//! [`ArrayKernels`] is a [`VortexSession`](vortex_session::VortexSession) variable that holds an
//! [`FnRegistry`] of kernel function pointers keyed by the identities of the arrays they operate
//! on. It is consulted by the optimizer during execution — before the child encoding's static
//! `PARENT_RULES` — so that a plugin can add a new rule or override a built-in without touching
//! the encoding's vtable. Entries for the parent-reduce kind are typed as [`ReduceParentFn`].
//!
//! The registry is empty by default. Downstream crates obtain [`ArrayKernels`] via
//! [`ArrayKernelsExt::kernels`] and register kernel function pointers through the typed helpers
//! like [`ArrayKernels::register_reduce_parent`].

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

/// Shared hasher used to combine `(outer, child, FnKind)` tuples into `FnRegistry` keys. A single
/// global instance is cheap and ensures all callers produce the same hash for the same tuple, so
/// lookups succeed across modules.
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

/// Kind tag mixed into a registry key so that the same `(outer, child)` encoding pair can hold
/// different kernel kinds (parent-reduce, reduce, execute, etc.) in one shared [`FnRegistry`]
/// without collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FnKind {
    Reduce,
    ReduceParent,
    ExecuteParent,
    Execute,
}

/// Session-scoped registry of pluggable kernel function pointers.
///
/// Entries are keyed by an `(outer_id, child_id, FnKind)` tuple that the typed `register_*` and
/// `find_*` helpers hash into the underlying [`FnRegistry`] key. Callers should always go through
/// the typed helpers rather than the raw registry so the hash scheme stays consistent.
#[derive(Debug, Default)]
pub struct ArrayKernels {
    registry: FnRegistry,
}

impl ArrayKernels {
    /// Create an empty [`ArrayKernels`] with no kernels registered.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Register a [`ReduceParentFn`] for `(outer, child)`, replacing any previously registered
    /// entry for the same pair.
    ///
    /// The optimizer will invoke `f` when it sees a parent with encoding id `outer` holding a
    /// child with encoding id `child` during a `reduce_parent` step, before trying the child
    /// encoding's static `PARENT_RULES`. `outer` is typically the parent's encoding id — for a
    /// `ScalarFnArray`, this is the scalar function's id (e.g. `Cast.id()`).
    pub fn register_reduce_parent(&self, outer: Id, child: Id, f: ReduceParentFn) {
        self.registry
            .register(self.hash_fn_ids(outer, child, FnKind::ReduceParent), f)
    }

    /// Look up the [`ReduceParentFn`] registered for `(outer, child)`.
    ///
    /// Returns `None` when no function is registered for this pair. The returned `Arc` is
    /// owned, so callers can drop the borrow on this [`ArrayKernels`] before invoking the
    /// function.
    pub fn find_reduce_parent(&self, outer: Id, child: Id) -> Option<Arc<ReduceParentFn>> {
        self.registry
            .find(self.hash_fn_ids(outer, child, FnKind::ReduceParent))
    }

    /// Return `true` if a [`ReduceParentFn`] is registered for `(outer, child)`.
    pub fn contains_reduce_parent(&self, outer: Id, child: Id) -> bool {
        self.registry
            .contains(self.hash_fn_ids(outer, child, FnKind::ReduceParent))
    }

    /// Combine a `(outer, child, fn_kind)` tuple into the `u64` key expected by the underlying
    /// [`FnRegistry`]. Using the shared [`FN_HASHER`] guarantees register and find produce the
    /// same key for the same logical pair.
    fn hash_fn_ids(&self, outer: Id, child: Id, fn_kind: FnKind) -> u64 {
        FN_HASHER.hash_one((outer, child, fn_kind))
    }
}

/// Extension trait for accessing [`ArrayKernels`] from a [`VortexSession`](vortex_session::VortexSession).
pub trait ArrayKernelsExt: SessionExt {
    /// Returns the [`ArrayKernels`] session variable, inserting a default-constructed one if
    /// none has been registered on the session yet.
    fn kernels(&self) -> Ref<'_, ArrayKernels> {
        self.get::<ArrayKernels>()
    }
}

impl<S: SessionExt> ArrayKernelsExt for S {}
