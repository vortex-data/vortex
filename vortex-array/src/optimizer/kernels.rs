// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry for optimizer kernels.
//!
//! [`ArrayKernels`] stores function pointers that participate in array optimization without
//! adding rules to an encoding vtable. The optimizer currently consults it for parent-reduce
//! rewrites before the child encoding's static `PARENT_RULES`. A registered function can
//! therefore add a rule for an extension encoding or take precedence over a built-in rule.
//!
//! Kernel entries are addressed by `(outer_id, child_id, kind)`. For parent-reduce kernels,
//! `outer_id` is the id returned by the parent array's `encoding_id()` and `child_id` is the
//! child array's `encoding_id()`. For [`ScalarFn`](crate::arrays::ScalarFn) parents, the parent
//! id is the scalar function id.
//!
//! Sessions created by the top-level `vortex` crate install an empty registry by default. Other
//! sessions can add it with [`VortexSession::with`](vortex_session::VortexSession::with) or rely
//! on [`ArrayKernelsExt::kernels`] to insert the default value.

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

/// Shared hasher used to combine `(outer, child, FnKind)` tuples into [`FnRegistry`] keys.
static FN_HASHER: LazyLock<DefaultHashBuilder> = LazyLock::new(DefaultHashBuilder::default);

/// Function pointer for a plugin-provided parent-reduce rewrite.
///
/// The optimizer calls this with the matched `child`, its `parent`, and the slot index where the
/// child appears. Return `Ok(Some(new_parent))` to replace the parent, or `Ok(None)` when the
/// rewrite does not apply.
///
/// Implementations must preserve the parent's logical length and dtype, matching the invariant
/// required of static parent-reduce rules.
pub type ReduceParentFn =
    fn(child: &ArrayRef, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

/// Disambiguates kernel kinds that share the same `(outer, child)` id pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(unused)]
enum FnKind {
    Reduce,
    ReduceParent,
    ExecuteParent,
    Execute,
}

/// Session-scoped registry of optimizer kernel functions.
///
/// Use the typed `register_*`, `find_*`, and `contains_*` methods rather than depending on the
/// internal hash format.
#[derive(Debug, Default)]
pub struct ArrayKernels {
    registry: FnRegistry,
}

impl ArrayKernels {
    /// Create an empty [`ArrayKernels`] with no kernels registered.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Register a [`ReduceParentFn`] for `(outer, child)`.
    ///
    /// The optimizer will invoke `f` when it sees a parent with encoding id `outer` holding a
    /// child with encoding id `child` during a `reduce_parent` step, before trying the child
    /// encoding's static `PARENT_RULES`. `outer` is usually the parent array's encoding id. For
    /// `ScalarFnArray`, it is the scalar function id, for example `Cast.id()`.
    ///
    /// Replaces any function already registered for the same pair.
    pub fn register_reduce_parent(&self, outer: Id, child: Id, f: ReduceParentFn) {
        self.registry
            .register(self.hash_fn_ids(outer, child, FnKind::ReduceParent), f)
    }

    /// Look up the [`ReduceParentFn`] registered for `(outer, child)`.
    ///
    /// Returns an owned [`Arc`] so the session-variable borrow can be dropped before invoking the
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

    /// Combine a typed kernel id tuple into the `u64` key expected by the underlying
    /// [`FnRegistry`]. All typed helpers use this path so registration and lookup agree.
    fn hash_fn_ids(&self, outer: Id, child: Id, fn_kind: FnKind) -> u64 {
        FN_HASHER.hash_one((outer, child, fn_kind))
    }
}

/// Extension trait for accessing optimizer kernels from a
/// [`VortexSession`](vortex_session::VortexSession).
pub trait ArrayKernelsExt: SessionExt {
    /// Returns the [`ArrayKernels`] session variable, inserting a default-constructed one if
    /// none has been registered on the session yet.
    fn kernels(&self) -> Ref<'_, ArrayKernels> {
        self.get::<ArrayKernels>()
    }
}

impl<S: SessionExt> ArrayKernelsExt for S {}
