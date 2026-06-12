// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry for optimizer kernels.
//!
//! [`ArrayKernels`] stores function pointers that participate in array optimization and execution
//! without adding rules or kernels to an encoding vtable. The optimizer consults it for
//! parent-reduce rewrites before the child encoding's static `PARENT_RULES`, and the executor
//! consults it for parent execution before the child encoding's static parent kernels. A
//! registered function can therefore add support for an extension encoding or take precedence over
//! a built-in rule or kernel. When several functions are registered for the same key and kind,
//! they are tried in registration order until one applies.
//!
//! Kernel entries are addressed by `(outer_id, child_id)`. For parent-reduce and execute-parent
//! kernels, `outer_id` is the id returned by the parent array's `encoding_id()` and `child_id` is
//! the child array's `encoding_id()`. For [`ScalarFn`](crate::arrays::ScalarFn) parents, the
//! parent id is the scalar function id.
//!
//! Because registered functions have different signatures for each kernel kind, the registry
//! maintains one storage map per function type rather than a single type-erased map.
//!
//! Sessions created by the top-level `vortex` crate install the default registry. Other sessions
//! can add it with [`VortexSession::with`](vortex_session::VortexSession::with) or rely on
//! [`ArrayKernelsExt::kernels`] to insert the default value.

use std::any::Any;
use std::borrow::Borrow;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Id;
use vortex_utils::aliases::DefaultHashBuilder;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arc_swap_map::ArcSwapMap;
use crate::array::VTable;
use crate::arrays::Struct;
use crate::arrays::struct_::compute::cast::struct_cast_execute_parent;
use crate::arrays::struct_::compute::rules::struct_cast_reduce_parent;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;

/// Shared hasher used to combine `(outer, child)` tuples into registry keys.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
struct ReduceParentFnId(u64);

impl From<u64> for ReduceParentFnId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl Borrow<u64> for ReduceParentFnId {
    fn borrow(&self) -> &u64 {
        &self.0
    }
}

/// Function pointer for a plugin-provided parent execution.
///
/// The executor calls this with the matched `child`, its `parent`, the slot index where the child
/// appears, and the current [`ExecutionCtx`]. Return `Ok(Some(new_parent))` to replace the parent
/// with an executed result, or `Ok(None)` when the kernel does not apply.
///
/// Implementations must preserve the parent's logical length and dtype, matching the invariant
/// required of static `execute_parent` kernels.
pub type ExecuteParentFn = fn(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
struct ExecuteParentFnId(u64);

impl From<u64> for ExecuteParentFnId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl Borrow<u64> for ExecuteParentFnId {
    fn borrow(&self) -> &u64 {
        &self.0
    }
}

/// Session-scoped registry of optimizer kernel functions.
///
/// Each kernel kind has its own storage map, keyed by `(outer_id, child_id)`. Registering
/// functions for an existing key appends them to that key's ordered list.
#[derive(Debug)]
pub struct ArrayKernels {
    reduce_parent: ArcSwapMap<ReduceParentFnId, Arc<[ReduceParentFn]>>,
    execute_parent: ArcSwapMap<ExecuteParentFnId, Arc<[ExecuteParentFn]>>,
}

impl Default for ArrayKernels {
    fn default() -> ArrayKernels {
        let this = Self::empty();
        this.register_builtin_reduce_parent();
        this.register_builtin_execute_parent();
        this
    }
}

impl ArrayKernels {
    /// Create an empty [`ArrayKernels`] with no kernels registered.
    pub fn empty() -> Self {
        Self {
            reduce_parent: ArcSwapMap::default(),
            execute_parent: ArcSwapMap::default(),
        }
    }

    fn register_builtin_reduce_parent(&self) {
        self.register_reduce_parent(
            Cast.id(),
            Struct.id(),
            &[struct_cast_reduce_parent as ReduceParentFn],
        );
    }

    fn register_builtin_execute_parent(&self) {
        self.register_execute_parent(
            Cast.id(),
            Struct.id(),
            &[struct_cast_execute_parent as ExecuteParentFn],
        );
    }

    /// Register [`ReduceParentFn`]s for `(parent, child)`.
    ///
    /// The optimizer invokes these functions in registration order when it sees a parent with
    /// encoding id `parent` holding a child with encoding id `child` during a `reduce_parent`
    /// step, before trying the child encoding's static `PARENT_RULES`. `parent` is usually the
    /// parent array's encoding id. For `ScalarFnArray`, it is the scalar function id, for example
    /// `Cast.id()`.
    ///
    /// If functions have already been registered for the same pair, these functions are appended
    /// after them.
    pub fn register_reduce_parent(&self, parent: Id, child: Id, fns: &[ReduceParentFn]) {
        self.reduce_parent
            .extend(hash_fn_id(parent, child).into(), fns);
    }

    /// Look up the [`ReduceParentFn`]s registered for `(parent, child)`.
    ///
    /// Returns an owned [`Arc`] so the session-variable borrow can be dropped before invoking the
    /// functions.
    pub fn find_reduce_parent(&self, parent: Id, child: Id) -> Option<Arc<[ReduceParentFn]>> {
        self.reduce_parent.get(&hash_fn_id(parent, child))
    }

    /// Register [`ExecuteParentFn`]s for `(parent, child)`.
    ///
    /// The executor invokes these functions in registration order when it sees a parent with
    /// encoding id `parent` holding a child with encoding id `child` during a parent execution
    /// step, before trying the child encoding's static parent kernels.
    ///
    /// If functions have already been registered for the same pair, these functions are appended
    /// after them.
    pub fn register_execute_parent(&self, parent: Id, child: Id, fns: &[ExecuteParentFn]) {
        self.execute_parent
            .extend(hash_fn_id(parent, child).into(), fns);
    }

    /// Look up the [`ExecuteParentFn`]s registered for `(parent, child)`.
    ///
    /// Returns an owned [`Arc`] so the session-variable borrow can be dropped before invoking the
    /// functions.
    pub fn find_execute_parent(&self, parent: Id, child: Id) -> Option<Arc<[ExecuteParentFn]>> {
        self.execute_parent.get(&hash_fn_id(parent, child))
    }
}

fn hash_fn_id(parent: Id, child: Id) -> u64 {
    FN_HASHER.hash_one((parent, child))
}

impl SessionVar for ArrayKernels {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
