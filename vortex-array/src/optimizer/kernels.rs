// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry for optimizer kernels.
//!
//! [`ArrayKernels`] stores the parent-reduce rules and parent-execution kernels that participate in
//! array optimization and execution without adding them to an encoding vtable. The optimizer
//! consults it for every parent-reduce rewrite, and the executor consults it for parent execution,
//! so a session can add support for an extension encoding or register a rule ahead of a built-in
//! one. When several entries are registered for the same key and kind, they are tried in
//! registration order until one applies.
//!
//! Kernel entries are addressed by `(outer_id, child_id)`. For parent-reduce and execute-parent
//! kernels, `outer_id` is the id returned by the parent array's `encoding_id()` and `child_id` is
//! the child array's `encoding_id()`. For [`ScalarFn`] parents, the
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
use std::any::TypeId;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::VTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::reduce_parent_with_kernels;
use crate::optimizer::rules::ArrayParentReduceRule;

/// Shared hasher used to combine `(outer, child)` tuples into registry keys.
static FN_HASHER: LazyLock<DefaultHashBuilder> = LazyLock::new(DefaultHashBuilder::default);

/// Type-erased parent-reduce kernel stored in the session registry.
pub(crate) trait DynReduceParentKernel: Debug + Send + Sync + 'static {
    /// Attempt to rewrite `parent` given its child at `child_idx`.
    fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub(crate) type ReduceParentKernelRef = Arc<dyn DynReduceParentKernel>;

/// A snapshot of the session's parent-reduce kernels, as the optimizer consults them.
#[derive(Clone, Debug)]
pub(crate) struct ParentReduceKernels {
    /// Kernels keyed by hashed `(parent, child)` pair.
    kernels: Arc<HashMap<ReduceParentFnId, Arc<[ReduceParentKernelRef]>>>,
    /// The child ids each parent id has kernels for, so a parent can skip children that no kernel
    /// is registered for without probing each pair.
    children: Arc<HashMap<Id, Arc<[Id]>>>,
}

impl ParentReduceKernels {
    /// Returns the kernels registered for `(parent, child)`.
    pub(crate) fn get(&self, parent: Id, child: Id) -> Option<&Arc<[ReduceParentKernelRef]>> {
        self.kernels.get(&hash_fn_id(parent, child))
    }

    /// Returns the child ids that kernels are registered for under `parent`.
    pub(crate) fn children(&self, parent: Id) -> Option<&[Id]> {
        self.children.get(&parent).map(AsRef::as_ref)
    }
}

struct RegisteredReduceParentRule<V, R> {
    rule: R,
    _child: PhantomData<V>,
}

impl<V, R: Debug> Debug for RegisteredReduceParentRule<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredReduceParentRule")
            .field("rule", &self.rule)
            .finish()
    }
}

impl<V, R> DynReduceParentKernel for RegisteredReduceParentRule<V, R>
where
    V: VTable,
    R: ArrayParentReduceRule<V>,
{
    fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(child) = child.as_opt::<V>() else {
            return Ok(None);
        };
        let Some(parent) = R::Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(child, parent, child_idx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub(crate) struct ReduceParentFnId(u64);

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

/// Registry of [`ReduceParentKernelRef`]s, keyed by hashed `(parent, child)` pair.
type ReduceParentRegistry = ArcSwapMap<ReduceParentFnId, Arc<[ReduceParentKernelRef]>>;
/// Registry of [`ExecuteParentKernelRef`]s, keyed by hashed `(parent, child)` pair.
type ExecuteParentRegistry = ArcSwapMap<ExecuteParentFnId, Arc<[ExecuteParentKernelRef]>>;

/// Session-scoped registry of optimizer kernel functions.
///
/// Each kernel kind has its own storage map, keyed by `(outer_id, child_id)`. Registering
/// functions for an existing key appends them to that key's ordered list.
#[derive(Clone, Debug)]
pub struct ArrayKernels {
    reduce_parent: ReduceParentRegistry,
    reduce_parent_children: ArcSwapMap<Id, Arc<[Id]>>,
    execute_parent: ExecuteParentRegistry,
}

impl Default for ArrayKernels {
    fn default() -> ArrayKernels {
        Self::empty()
    }
}

impl ArrayKernels {
    /// Create an empty [`ArrayKernels`] with no kernels registered.
    pub fn empty() -> Self {
        Self {
            reduce_parent: ReduceParentRegistry::default(),
            reduce_parent_children: ArcSwapMap::default(),
            execute_parent: ExecuteParentRegistry::default(),
        }
    }

    /// Register an [`ArrayParentReduceRule`] for a `child` encoding under `parent`.
    ///
    /// The optimizer tries the rules registered for a pair in registration order when it sees a
    /// parent with encoding id `parent` holding a child of the `child` encoding. The child id comes
    /// from the vtable that the rule applies to, so it always matches the rule. The parent id is
    /// an encoding id, except that a `ScalarFnArray` is identified by its scalar function id, for
    /// example `Cast.id()`, or by [`AnyScalarFn::id`] for a rule that applies to any scalar
    /// function array. Rules for the exact ids are tried before those registered under
    /// [`AnyScalarFn::id`].
    ///
    /// Use [`Self::register_scalar_fn_reduce_parent_rule`] for a rule on a scalar function child.
    pub fn register_reduce_parent_rule<V, R>(&self, parent: Id, child: V, rule: R)
    where
        V: VTable,
        R: ArrayParentReduceRule<V>,
    {
        debug_assert!(
            TypeId::of::<V>() != TypeId::of::<ScalarFn>(),
            "register scalar function children with register_scalar_fn_reduce_parent_rule"
        );
        self.push_reduce_parent_rule::<V, R>(parent, child.id(), rule);
    }

    /// Register an [`ArrayParentReduceRule`] for a scalar function child, of any function, under
    /// `parent`.
    ///
    /// The rule is keyed by [`AnyScalarFn::id`] on the child side. See
    /// [`Self::register_reduce_parent_rule`] for how `parent` is identified.
    pub fn register_scalar_fn_reduce_parent_rule<R>(&self, parent: Id, rule: R)
    where
        R: ArrayParentReduceRule<ScalarFn>,
    {
        self.push_reduce_parent_rule::<ScalarFn, R>(parent, AnyScalarFn.id(), rule);
    }

    fn push_reduce_parent_rule<V, R>(&self, parent: Id, child: Id, rule: R)
    where
        V: VTable,
        R: ArrayParentReduceRule<V>,
    {
        self.reduce_parent.push(
            hash_fn_id(parent, child).into(),
            Arc::new(RegisteredReduceParentRule {
                rule,
                _child: PhantomData::<V>,
            }) as ReduceParentKernelRef,
        );
        self.record_reduce_parent_child(parent, child);
    }

    /// Record that `parent` has parent-reduce kernels for `child`, so the optimizer probes only the
    /// children of a parent that kernels exist for, as the execute-parent marker does for parents.
    fn record_reduce_parent_child(&self, parent: Id, child: Id) {
        if self.reduce_parent_children.read(|children| {
            children
                .get(&parent)
                .is_some_and(|ids| ids.contains(&child))
        }) {
            return;
        }
        self.reduce_parent_children.extend(parent, &[child]);
    }

    /// Try the parent-reduce kernels registered for `parent` and `child` once, without
    /// optimizing the result.
    pub fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        reduce_parent_with_kernels(&self.reduce_parent_snapshot(), parent, child, child_idx)
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
        self.mark_execute_parent(parent);
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
        self.mark_execute_parent(parent);
    }

    /// Record that `parent` owns at least one execute-parent kernel.
    ///
    /// The marker lives in the same map as the kernels, under a key no `(parent, child)` pair
    /// hashes to, so the executor's snapshot answers "does this parent have any kernel?" with
    /// one probe and no extra state. The marker's kernel list is empty.
    fn mark_execute_parent(&self, parent: Id) {
        self.execute_parent
            .insert_if_absent(execute_parent_marker_key(parent).into(), Arc::from([]));
    }

    /// Returns true when one or more execute-parent kernels are registered for `(parent, child)`.
    pub fn has_execute_parent(&self, parent: Id, child: Id) -> bool {
        self.execute_parent
            .get(&hash_fn_id(parent, child))
            .is_some()
    }

    /// Return the currently published parent-reduce kernel snapshot.
    pub(crate) fn reduce_parent_snapshot(&self) -> ParentReduceKernels {
        ParentReduceKernels {
            kernels: self.reduce_parent.snapshot(),
            children: self.reduce_parent_children.snapshot(),
        }
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

/// Return the registry key under which `parent` is marked as owning execute-parent kernels.
///
/// Hashed from a different tuple type than the `(Id, Id)` pair keys, so it cannot equal a pair
/// key by construction, only by a 64-bit collision; either way a hit is harmless, as the marker's
/// kernel list is empty and the prefilter only decides whether to probe the pairs.
pub(crate) fn execute_parent_marker_key(parent: Id) -> u64 {
    FN_HASHER.hash_one(("execute_parent", parent))
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

/// Derefs to the held [`ArrayKernels`] registry, so a [`KernelSession`] (or a
/// [`SessionGuard<KernelSession>`](SessionGuard) read from a session) can be used wherever an
/// `&ArrayKernels` is expected.
impl Deref for KernelSession {
    type Target = ArrayKernels;

    fn deref(&self) -> &ArrayKernels {
        &self.kernels
    }
}

impl Default for KernelSession {
    fn default() -> Self {
        // The built-in parent-reduce rules and execute-parent kernels are registered by the
        // per-encoding `initialize` functions, which operate on a session. `KernelSession` clones
        // share their registry storage, so kernels registered into the temporary session land in
        // `this.kernels`.
        let this = Self {
            kernels: ArrayKernels::empty(),
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
    /// Returns the session's [`KernelSession`], inserting a default one (with the built-in
    /// kernels) if it does not exist.
    ///
    /// The returned [`SessionGuard`] borrows the session snapshot it was read from (so the registry
    /// stays alive even if the session is concurrently mutated) and derefs through [`KernelSession`]
    /// to the [`ArrayKernels`] registry, so it can be used wherever an `&ArrayKernels` is expected.
    /// The registry shares its storage with the session, so kernels registered through it remain
    /// visible to the session.
    fn kernels(&self) -> SessionGuard<'_, KernelSession> {
        self.get::<KernelSession>()
    }
}

impl<S: SessionExt> ArrayKernelsExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    use super::ArrayKernelsExt;
    use super::KernelSession;
    use crate::ArrayVTable;
    use crate::arrays::Bool;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::binary::Binary;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

    #[test]
    fn kernel_session_default_registers_builtin_kernels() {
        assert!(SESSION.kernels().has_execute_parent(Binary.id(), Bool.id()));
    }

    #[test]
    fn initialize_registers_builtin_kernels_into_empty_kernel_session() {
        let session = VortexSession::empty().with_some(KernelSession::empty());

        assert!(!session.kernels().has_execute_parent(Binary.id(), Bool.id()));

        crate::initialize(&session);

        assert!(session.kernels().has_execute_parent(Binary.id(), Bool.id()));
    }

    #[test]
    fn kernels_inserts_default_kernel_session() {
        let session = VortexSession::empty();

        // `kernels()` uses `get`, so it inserts a default `KernelSession` (with the built-in
        // kernels) rather than returning `None`.
        assert!(session.kernels().has_execute_parent(Binary.id(), Bool.id()));
    }
}
