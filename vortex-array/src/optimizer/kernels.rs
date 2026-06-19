// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry for optimizer kernels.
//!
//! [`ArrayKernels`] stores function pointers that participate in array optimization and execution
//! without adding rules or kernels to an encoding vtable. The optimizer consults it for
//! parent-reduce rewrites before the child encoding's static `PARENT_RULES`, and the executor
//! consults it for parent execution. A registered function can therefore add support for an
//! extension encoding or take precedence over a built-in rule. When several functions are
//! registered for the same key and kind, they are tried in registration order until one applies.
//!
//! Kernel entries are addressed by `(outer_id, child_id)`. For parent-reduce and execute-parent
//! kernels, `outer_id` is the id returned by the parent array's `encoding_id()` and `child_id` is
//! the child array's `encoding_id()`. For [`ScalarFn`](crate::arrays::ScalarFn) parents, the
//! parent id is the scalar function id.
//!
//! Because registered functions have different signatures for each kernel kind, the registry
//! maintains one storage map per function type rather than a single type-erased map.
//!
//! [`KernelSession`] is the session variable that owns this registry. Its [`Default`]
//! implementation installs vortex-array's built-in parent-reduce and execute-parent kernels, so a
//! session built with [`KernelSession`] participates in the same optimizations and fused execution
//! as the built-in encodings.

use std::any::Any;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arc_swap_map::ArcSwapMap;
use crate::array::VTable;
use crate::arrays::Struct;
use crate::arrays::struct_::compute::rules::struct_cast_reduce_parent;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
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

/// Type-erased execute-parent kernel stored in the session registry.
pub trait DynExecuteParentKernel: Debug + Send + Sync + 'static {
    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub(crate) type ExecuteParentKernelRef = Arc<dyn DynExecuteParentKernel>;

pub(crate) type ParentExecutionKernels = HashMap<ExecuteParentFnId, Arc<[ExecuteParentKernelRef]>>;

#[derive(Debug)]
struct ExecuteParentFnKernel(ExecuteParentFn);

impl DynExecuteParentKernel for ExecuteParentFnKernel {
    fn execute_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        self.0(child, parent, child_idx, ctx)
    }
}

#[derive(Debug)]
struct RegisteredExecuteParentKernel<V, K> {
    _child: V,
    kernel: K,
}

impl<V, K> DynExecuteParentKernel for RegisteredExecuteParentKernel<V, K>
where
    V: VTable,
    K: ExecuteParentKernel<V>,
{
    fn execute_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(child) = child.as_opt::<V>() else {
            return Ok(None);
        };
        let Some(parent) = K::Parent::try_match(parent) else {
            return Ok(None);
        };

        self.kernel.execute_parent(child, parent, child_idx, ctx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub(crate) struct ExecuteParentFnId(u64);

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
#[derive(Clone, Debug)]
pub struct ArrayKernels {
    reduce_parent: ArcSwapMap<ReduceParentFnId, Arc<[ReduceParentFn]>>,
    execute_parent: ArcSwapMap<ExecuteParentFnId, Arc<[ExecuteParentKernelRef]>>,
}

impl Default for ArrayKernels {
    fn default() -> ArrayKernels {
        let this = Self::empty();
        this.register_builtin_reduce_parent();
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
    /// step.
    ///
    /// If functions have already been registered for the same pair, these functions are appended
    /// after them.
    pub fn register_execute_parent(&self, parent: Id, child: Id, fns: &[ExecuteParentFn]) {
        let kernels: Vec<ExecuteParentKernelRef> = fns
            .iter()
            .map(|f| Arc::new(ExecuteParentFnKernel(*f)) as ExecuteParentKernelRef)
            .collect();
        self.execute_parent
            .extend(hash_fn_id(parent, child).into(), kernels.as_slice());
    }

    /// Register a typed [`ExecuteParentKernel`] for `(parent, child.id())`.
    ///
    /// The executor invokes registered kernels in registration order before falling through to
    /// later registered kernels for the same key. `parent` is usually the parent array's encoding
    /// id. For `ScalarFnArray`, it is the scalar function id, for example `Cast.id()`.
    ///
    /// If kernels have already been registered for the same pair, this kernel is appended after
    /// them; registering for an existing key cannot override built-in kernels installed earlier.
    pub fn register_execute_parent_kernel<V, K>(&self, parent: Id, child: V, kernel: K)
    where
        V: VTable,
        K: ExecuteParentKernel<V>,
    {
        let child_id = child.id();
        self.execute_parent.push(
            hash_fn_id(parent, child_id).into(),
            Arc::new(RegisteredExecuteParentKernel {
                _child: child,
                kernel,
            }) as ExecuteParentKernelRef,
        );
    }

    /// Returns true when one or more execute-parent kernels are registered for `(parent, child)`.
    pub fn has_execute_parent(&self, parent: Id, child: Id) -> bool {
        self.execute_parent
            .get(&hash_fn_id(parent, child))
            .is_some()
    }

    /// Return the currently published execute-parent kernel snapshot.
    pub(crate) fn execute_parent_snapshot(&self) -> Arc<ParentExecutionKernels> {
        self.execute_parent.snapshot()
    }
}

fn hash_fn_id(parent: Id, child: Id) -> u64 {
    FN_HASHER.hash_one((parent, child))
}

/// Return the registry key for execute-parent kernels registered for `(parent, child)`.
pub(crate) fn execute_parent_key(parent: Id, child: Id) -> u64 {
    hash_fn_id(parent, child)
}

/// Session-scoped holder for the optimizer kernel registry.
///
/// `KernelSession` is the session variable that owns an [`ArrayKernels`] registry. Its [`Default`]
/// implementation installs vortex-array's built-in parent-reduce and execute-parent kernels,
/// mirroring how [`ScalarFnSession`](crate::scalar_fn::session::ScalarFnSession) and the other
/// session variables register their built-ins.
#[derive(Clone, Debug)]
pub struct KernelSession {
    kernels: ArrayKernels,
}

impl KernelSession {
    /// Create a [`KernelSession`] with an empty kernel registry.
    pub fn empty() -> Self {
        Self {
            kernels: ArrayKernels::empty(),
        }
    }

    /// Returns the [`ArrayKernels`] registry held by this session.
    pub fn kernels(&self) -> &ArrayKernels {
        &self.kernels
    }
}

impl Default for KernelSession {
    fn default() -> Self {
        // `ArrayKernels::default` installs the built-in parent-reduce kernels. The execute-parent
        // kernels are registered by the per-encoding `initialize` functions, which operate on a
        // session. `KernelSession` clones share their registry storage, so kernels registered into
        // the temporary session land in `this.kernels`.
        let this = Self {
            kernels: ArrayKernels::default(),
        };
        let session = VortexSession::empty().with_some(this.clone());
        crate::arrays::initialize(&session);
        this
    }
}

impl SessionVar for KernelSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing the optimizer kernel registry from a [`VortexSession`].
pub trait ArrayKernelsExt: SessionExt {
    /// Returns the active [`ArrayKernels`] registry if the session contains a [`KernelSession`].
    fn kernels_opt(&self) -> Option<&ArrayKernels> {
        self.get_opt::<KernelSession>().map(KernelSession::kernels)
    }

    /// Returns the active [`ArrayKernels`] registry.
    ///
    /// Panics if the session does not contain a [`KernelSession`].
    fn kernels(&self) -> &ArrayKernels {
        self.kernels_opt()
            .unwrap_or_else(|| vortex_panic!("Session does not contain a KernelSession"))
    }
}

impl<S: SessionExt> ArrayKernelsExt for S {}

#[cfg(test)]
mod tests {
    use vortex_session::VortexSession;

    use super::ArrayKernelsExt;
    use super::KernelSession;
    use crate::ArrayVTable;
    use crate::arrays::Bool;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::binary::Binary;

    #[test]
    fn kernel_session_default_registers_builtin_kernels() {
        let session = VortexSession::empty().with::<KernelSession>();

        assert!(session.kernels().has_execute_parent(Binary.id(), Bool.id()));
    }

    #[test]
    fn initialize_registers_builtin_kernels_into_empty_kernel_session() {
        let session = VortexSession::empty().with_some(KernelSession::empty());

        assert!(!session.kernels().has_execute_parent(Binary.id(), Bool.id()));

        crate::initialize(&session);

        assert!(session.kernels().has_execute_parent(Binary.id(), Bool.id()));
    }

    #[test]
    fn kernels_opt_is_none_without_kernel_session() {
        let session = VortexSession::empty();

        assert!(session.kernels_opt().is_none());
    }
}
