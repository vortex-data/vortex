// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of parent kernels.
//!
//! [`ArrayKernels`] holds two kinds of kernels. Each kernel is registered for a parent and a child:
//!
//! - The optimizer uses parent-reduce rules to rewrite a parent array from one of its children,
//!   without decoding data.
//! - The executor uses execute-parent kernels to execute a parent array through one of its
//!   children.
//!
//! The parent is the encoding of the parent array, or the function of a scalar function array.
//! When more than one kernel is registered for the same parent and child, they are tried in
//! registration order until one applies.
//!
//! [`KernelSession`] is the session variable that holds the registry. Its [`Default`]
//! implementation registers the built-in kernels of `vortex-array`.

use std::any::Any;
use std::any::TypeId;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;

use arc_swap::ArcSwap;
use vortex_error::VortexResult;
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
use crate::scalar_fn::ScalarFnVTable;

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

pub(crate) type ReduceParentKernelRef = &'static dyn DynReduceParentKernel;

/// A snapshot of the session's parent-reduce kernels, as the optimizer consults them.
#[derive(Clone, Debug, Default)]
pub(crate) struct ParentReduceKernels {
    /// Kernels keyed by hashed `(parent, child)` pair.
    kernels: HashMap<ReduceParentFnId, Box<[ReduceParentKernelRef]>>,
    /// The child ids each parent id has kernels for, so a parent can skip children that no kernel
    /// is registered for without probing each pair.
    children: HashMap<Id, Box<[Id]>>,
}

impl ParentReduceKernels {
    /// Returns the kernels registered for `(parent, child)`.
    pub(crate) fn get(&self, parent: Id, child: Id) -> Option<&[ReduceParentKernelRef]> {
        self.kernels
            .get(&hash_fn_id(parent, child))
            .map(AsRef::as_ref)
    }

    /// Returns the child ids that kernels are registered for under `parent`.
    pub(crate) fn children(&self, parent: Id) -> Option<&[Id]> {
        self.children.get(&parent).map(AsRef::as_ref)
    }

    fn push(&mut self, parent: Id, child: Id, kernel: ReduceParentKernelRef) {
        append(
            self.kernels
                .entry(hash_fn_id(parent, child).into())
                .or_default(),
            [kernel],
        );

        let children = self.children.entry(parent).or_default();
        if !children.contains(&child) {
            append(children, [child]);
        }
    }
}

/// Appends `items` to a boxed slice. Registration replaces the kernel maps rather than growing
/// them in place, so the slices keep no spare capacity.
fn append<T>(slice: &mut Box<[T]>, items: impl IntoIterator<Item = T>) {
    let mut items_vec = std::mem::take(slice).into_vec();
    items_vec.extend(items);
    *slice = items_vec.into_boxed_slice();
}

#[repr(transparent)]
struct RegisteredReduceParentRule<V, R> {
    rule: R,
    _child: PhantomData<V>,
}

impl<V: VTable, R: ArrayParentReduceRule<V>> RegisteredReduceParentRule<V, R> {
    /// Views a static rule as a parent-reduce kernel for the `V` child encoding.
    const fn lift(rule: &'static R) -> ReduceParentKernelRef {
        // SAFETY: `RegisteredReduceParentRule` is `repr(transparent)` over `rule`, and its only
        // other field is a zero-sized `PhantomData`, so it has the same layout as `R`.
        unsafe { &*ptr::from_ref(rule).cast::<Self>() }
    }
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
        let (Some(child), Some(parent)) = (child.as_opt::<V>(), R::Parent::try_match(parent))
        else {
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

pub(crate) type ParentExecutionKernels = HashMap<ExecuteParentFnId, Box<[ExecuteParentKernelRef]>>;

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

/// The parent that a parent-reduce rule is registered under.
///
/// A parent is an array encoding, such as [`Filter`](crate::arrays::Filter), or a scalar function,
/// such as [`Cast`](crate::scalar_fn::fns::cast::Cast), because a `ScalarFnArray` is identified by
/// its function. [`AnyScalarFn`] is the parent for a rule that applies to any scalar function
/// array. The `Kind` parameter only keeps these implementations apart, and callers never name it.
pub trait ReduceParent<Kind> {
    /// The id that the parent is registered under.
    fn parent_id(&self) -> Id;
}

/// The [`ReduceParent`] kind of an array encoding.
#[derive(Debug)]
pub struct EncodingParent;

/// The [`ReduceParent`] kind of a scalar function.
#[derive(Debug)]
pub struct ScalarFnParent;

impl<V: VTable> ReduceParent<EncodingParent> for V {
    fn parent_id(&self) -> Id {
        VTable::id(self)
    }
}

impl<F: ScalarFnVTable> ReduceParent<ScalarFnParent> for F {
    fn parent_id(&self) -> Id {
        ScalarFnVTable::id(self)
    }
}

impl ReduceParent<AnyScalarFn> for AnyScalarFn {
    fn parent_id(&self) -> Id {
        self.id()
    }
}

/// Session-scoped registry of parent-reduce rules and execute-parent kernels.
///
/// Kernels registered for the same parent and child are tried in registration order.
#[derive(Clone, Debug)]
pub struct ArrayKernels {
    reduce_parent: Arc<ArcSwap<ParentReduceKernels>>,
    execute_parent: Arc<ArcSwap<ParentExecutionKernels>>,
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
            reduce_parent: Arc::default(),
            execute_parent: Arc::default(),
        }
    }

    /// Register an [`ArrayParentReduceRule`] for a `child` encoding under `parent`.
    ///
    /// The optimizer tries the rules registered for a pair in registration order when it sees a
    /// `parent` holding a child of the `child` encoding. The child id comes from the vtable that
    /// the rule applies to, so it always matches the rule. The parent is an encoding, such as
    /// `Filter`, or a scalar function, such as `Cast`, or [`AnyScalarFn`] for a rule that applies
    /// to any scalar function array. Rules for the exact parent are tried before those registered
    /// under [`AnyScalarFn`].
    ///
    /// Use [`Self::register_scalar_fn_reduce_parent_rule`] for a rule on a scalar function child.
    ///
    /// Pass the rule as `const { &rule }` so that it lives in static memory.
    pub fn register_reduce_parent_rule<P, Kind, V, R>(&self, parent: P, child: V, rule: &'static R)
    where
        P: ReduceParent<Kind>,
        V: VTable,
        R: ArrayParentReduceRule<V>,
    {
        debug_assert!(
            TypeId::of::<V>() != TypeId::of::<ScalarFn>(),
            "register scalar function children with register_scalar_fn_reduce_parent_rule"
        );
        self.push_reduce_parent_rule::<V, R>(parent.parent_id(), child.id(), rule);
    }

    /// Register an [`ArrayParentReduceRule`] for a scalar function child, of any function, under
    /// `parent`.
    ///
    /// The rule is keyed by [`AnyScalarFn::id`] on the child side. See
    /// [`Self::register_reduce_parent_rule`] for how `parent` is identified and how to pass `rule`.
    pub fn register_scalar_fn_reduce_parent_rule<P, Kind, R>(&self, parent: P, rule: &'static R)
    where
        P: ReduceParent<Kind>,
        R: ArrayParentReduceRule<ScalarFn>,
    {
        self.push_reduce_parent_rule::<ScalarFn, R>(parent.parent_id(), AnyScalarFn.id(), rule);
    }

    fn push_reduce_parent_rule<V, R>(&self, parent: Id, child: Id, rule: &'static R)
    where
        V: VTable,
        R: ArrayParentReduceRule<V>,
    {
        let kernel = RegisteredReduceParentRule::<V, R>::lift(rule);

        self.reduce_parent.rcu(|current| {
            let mut next = ParentReduceKernels::clone(current);
            next.push(parent, child, kernel);
            next
        });
    }

    /// Try the parent-reduce kernels registered for `parent` and `child` once, without
    /// optimizing the result.
    pub fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        self.with_reduce_parent_kernels(|kernels| {
            reduce_parent_with_kernels(kernels, parent, child, child_idx)
        })
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
        self.push_execute_parent(parent, child, &kernels);
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
        let kernel: ExecuteParentKernelRef = Arc::new(RegisteredExecuteParentKernel {
            _child: child,
            kernel,
        });
        self.push_execute_parent(parent, child_id, &[kernel]);
    }

    /// Append `kernels` for `(parent, child)` and record that `parent` owns execute-parent kernels.
    ///
    /// The marker lives in the same map as the kernels, under a key no `(parent, child)` pair
    /// hashes to, so the executor's snapshot answers "does this parent have any kernel?" with
    /// one probe and no extra state. The marker's kernel list is empty.
    fn push_execute_parent(&self, parent: Id, child: Id, kernels: &[ExecuteParentKernelRef]) {
        self.execute_parent.rcu(|current| {
            let mut next = ParentExecutionKernels::clone(current);
            append(
                next.entry(hash_fn_id(parent, child).into()).or_default(),
                kernels.iter().cloned(),
            );
            next.entry(execute_parent_marker_key(parent).into())
                .or_default();
            next
        });
    }

    /// Returns true when one or more execute-parent kernels are registered for `(parent, child)`.
    pub fn has_execute_parent(&self, parent: Id, child: Id) -> bool {
        self.execute_parent
            .load()
            .contains_key(&hash_fn_id(parent, child))
    }

    /// Return the currently published parent-reduce kernel snapshot.
    pub(crate) fn reduce_parent_snapshot(&self) -> Arc<ParentReduceKernels> {
        self.reduce_parent.load_full()
    }

    /// Passes the currently published parent-reduce kernels to `f` without copying them.
    pub(crate) fn with_reduce_parent_kernels<R>(
        &self,
        f: impl FnOnce(&ParentReduceKernels) -> R,
    ) -> R {
        f(&self.reduce_parent.load())
    }

    /// Return the currently published execute-parent kernel snapshot.
    pub(crate) fn execute_parent_snapshot(&self) -> Arc<ParentExecutionKernels> {
        self.execute_parent.load_full()
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
