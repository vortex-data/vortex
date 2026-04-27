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

use arc_swap::ArcSwap;
use vortex_error::VortexResult;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Id;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;

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

/// Session-scoped registry of optimizer kernel functions.
#[derive(Debug, Default)]
pub struct ArrayKernels {
    reduce_parent: ArcSwap<HashMap<u64, Arc<[ReduceParentFn]>>>,
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
    pub fn register_reduce_parent<I: IntoIterator<Item = ReduceParentFn>>(
        &self,
        parent: Id,
        child: Id,
        fns: I,
    ) {
        let registry = self.reduce_parent.load();
        let id = self.hash_fn_ids(parent, child);
        let mut owned_registry = registry.as_ref().clone();
        if let Some(existing) = owned_registry.remove(&id) {
            owned_registry.insert(id, existing.as_ref().iter().cloned().chain(fns).collect());
        } else {
            owned_registry.insert(id, fns.into_iter().collect());
        }
        self.reduce_parent.store(Arc::new(owned_registry));
    }

    /// Look up the [`ReduceParentFn`] registered for `(outer, child)`.
    ///
    /// Returns an owned [`Arc`] so the session-variable borrow can be dropped before invoking the
    /// function.
    pub fn find_reduce_parent(&self, parent: Id, child: Id) -> Option<Arc<[ReduceParentFn]>> {
        let id = self.hash_fn_ids(parent, child);
        let map = self.reduce_parent.load();
        let entry = map.get(&id)?;
        Some(Arc::clone(entry))
    }

    /// Combine a typed kernel id tuple into the `u64` key expected by the underlying
    /// [`FnRegistry`]. All typed helpers use this path so registration and lookup agree.
    fn hash_fn_ids(&self, parent: Id, child: Id) -> u64 {
        FN_HASHER.hash_one((parent, child))
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
