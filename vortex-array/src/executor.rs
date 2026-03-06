// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env::VarError;
use std::fmt;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;
use std::sync::atomic::AtomicUsize;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::IntoArray;

/// Returns a stable identity key for an [`ArrayRef`] based on its `Arc` pointer address.
pub(crate) fn ptr_id(array: &ArrayRef) -> usize {
    Arc::as_ptr(array) as *const () as usize
}

/// Cache for deduplicating shared sub-expression execution.
///
/// When the same `Arc<dyn Array>` appears as a child of multiple parents (e.g., `a` in
/// `a < 10 & a > 5`), the first parent's execution populates the cache. Subsequent parents
/// skip directly to the most-progressed result.
///
/// Uses two maps:
/// - **forward**: original source pointer → `(Weak, ArrayRef)` where the `Weak` validates
///   the source is still alive (guards against allocator pointer recycling)
/// - **inverse**: current result pointer → original source pointer (integer keys only,
///   no strong refs to intermediates)
struct ExecutionCache {
    forward: HashMap<usize, (Weak<dyn DynArray>, ArrayRef)>,
    inverse: HashMap<usize, usize>,
}

impl ExecutionCache {
    fn new() -> Self {
        Self {
            forward: HashMap::new(),
            inverse: HashMap::new(),
        }
    }
}

/// Maximum number of iterations to attempt when executing an array before giving up and returning
/// an error.
pub(crate) static MAX_ITERATIONS: LazyLock<usize> =
    LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
        Ok(val) => val
            .parse::<usize>()
            .unwrap_or_else(|e| vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")),
        Err(VarError::NotPresent) => 128,
        Err(VarError::NotUnicode(_)) => {
            vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid unicode string")
        }
    });

/// Whether to enable the execution cache for shared sub-expressions.
///
/// When enabled, arrays that are referenced by multiple parents (i.e. `Arc::strong_count > 1`)
/// will have their execution results cached so that subsequent references skip redundant work.
/// Controlled by the `VORTEX_EXECUTION_CACHE` environment variable (`false`/`0` to disable).
///
/// Defaults to enabled.
static EXECUTION_CACHE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| match std::env::var("VORTEX_EXECUTION_CACHE") {
        Ok(val) => !matches!(val.as_str(), "0" | "false" | "FALSE" | "False"),
        Err(_) => true,
    });

/// Marker trait for types that an [`ArrayRef`] can be executed into.
///
/// Implementors must provide an implementation of `execute` that takes
/// an [`ArrayRef`] and an [`ExecutionCtx`], and produces an instance of the
/// implementor type.
///
/// Users should use the `Array::execute` or `Array::execute_as` methods
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}

impl dyn DynArray + '_ {
    /// Execute this array to produce an instance of `E`.
    ///
    /// See the [`Executable`] implementation for details on how this execution is performed.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        E::execute(self, ctx)
    }

    /// Execute this array, labeling the execution step with a name for tracing.
    pub fn execute_as<E: Executable>(
        self: Arc<Self>,
        _name: &'static str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<E> {
        E::execute(self, ctx)
    }
}

/// Execution context for batch CPU compute.
///
/// Accumulates a trace of execution steps. Individual steps are logged at TRACE level for
/// real-time following, and the full trace is dumped at DEBUG level when the context is dropped.
pub struct ExecutionCtx {
    id: usize,
    session: VortexSession,
    ops: Vec<String>,
    cache: Option<ExecutionCache>,
    cache_hits: usize,
    cache_misses: usize,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        static EXEC_CTX_ID: AtomicUsize = AtomicUsize::new(0);
        let id = EXEC_CTX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            id,
            session,
            ops: Vec::new(),
            cache: (*EXECUTION_CACHE_ENABLED).then(ExecutionCache::new),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Returns the original source pointer for this array if it should be cached.
    ///
    /// An array is cacheable if it is already tracked as an intermediate result in the
    /// inverse map, or if it is shared (`Arc::strong_count > 1`).
    pub(crate) fn cache_source_ptr(&self, array: &ArrayRef) -> Option<usize> {
        let cache = self.cache.as_ref()?;
        let id = ptr_id(array);
        if cache.inverse.contains_key(&id) {
            Some(*cache.inverse.get(&id).vortex_expect("just checked"))
        } else if Arc::strong_count(array) > 1 {
            Some(id)
        } else {
            None
        }
    }

    /// Attempts to get a cached result more progressed than `current_id`.
    pub(crate) fn cache_get(&mut self, source_ptr: usize, current_id: usize) -> Option<ArrayRef> {
        let (source_weak, cached) = self.cache.as_ref()?.forward.get(&source_ptr)?;
        // Guard against allocator pointer recycling: if the original source was dropped,
        // a new array may have been allocated at the same address. The Weak ref detects this.
        if source_weak.strong_count() == 0 {
            return None;
        }
        let result = (ptr_id(cached) != current_id).then(|| cached.clone());
        if result.is_some() {
            self.cache_hits += 1;
        }
        result
    }

    /// Records a single execution step in the cache.
    ///
    /// `source_weak` is a [`Weak`] reference to the original source array, used to detect
    /// when the source is dropped and its pointer potentially recycled by the allocator.
    pub(crate) fn cache_put(
        &mut self,
        source_ptr: usize,
        source_weak: &Weak<dyn DynArray>,
        result: &ArrayRef,
    ) {
        if let Some(cache) = self.cache.as_mut() {
            cache
                .forward
                .insert(source_ptr, (source_weak.clone(), result.clone()));
            cache.inverse.insert(ptr_id(result), source_ptr);
        }
    }

    /// Returns the number of cache hits during this execution.
    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    /// Returns the number of cache misses during this execution.
    pub fn cache_misses(&self) -> usize {
        self.cache_misses
    }

    /// Returns the number of entries currently stored in the cache.
    pub fn cache_size(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.forward.len())
    }

    /// Log an execution step at the current depth.
    ///
    /// Steps are accumulated and dumped as a single trace on Drop at DEBUG level.
    /// Individual steps are also logged at TRACE level for real-time following.
    ///
    /// Use the [`format_args!`] macro to create the `msg` argument.
    pub fn log(&mut self, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let formatted = format!(" - {msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
        }
    }
}

impl Display for ExecutionCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exec[{}]", self.id)
    }
}

impl Drop for ExecutionCtx {
    fn drop(&mut self) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            if !self.ops.is_empty() {
                // Unlike itertools `.format()` (panics in 0.14 on second format)
                struct FmtOps<'a>(&'a [String]);
                impl Display for FmtOps<'_> {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        for (i, op) in self.0.iter().enumerate() {
                            if i > 0 {
                                f.write_str("\n")?;
                            }
                            f.write_str(op)?;
                        }
                        Ok(())
                    }
                }
                tracing::debug!("exec[{}] trace:\n{}", self.id, FmtOps(&self.ops));
            }

            if self.cache.is_some() {
                tracing::debug!(
                    "exec[{}] cache: {} hits, {} misses, {} entries",
                    self.id,
                    self.cache_hits,
                    self.cache_misses,
                    self.cache_size(),
                );
            }
        }
    }
}

/// Executing an [`ArrayRef`] into an [`ArrayRef`] is the atomic execution loop within Vortex.
///
/// It attempts to take the smallest possible step of execution such that the returned array
/// is incrementally more "executed" than the input array. In other words, it is closer to becoming
/// a canonical array.
///
/// The execution steps are as follows:
/// 0. Check for canonical.
/// 1. Attempt to call `reduce_parent` on each child.
/// 2. Attempt to `reduce` the array with metadata-only optimizations.
/// 3. Attempt to call `execute_parent` on each child.
/// 4. Call `execute` on the array itself.
///
/// Most users will not call this method directly, instead preferring to specify an executable
/// target such as [`crate::Columnar`], [`Canonical`], or any of the canonical array types (such as
/// [`crate::arrays::PrimitiveArray`]).
impl Executable for ArrayRef {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let source_ptr = ctx.cache_source_ptr(&array);
        let array_id = ptr_id(&array);

        // Save a Weak ref before the array is moved into execute_array_step.
        // This is used to detect allocator pointer recycling in cache_put/cache_get.
        let source_weak = source_ptr.map(|_| Arc::downgrade(&array));

        // Check cache for a more-progressed result.
        if let Some(sp) = source_ptr {
            if let Some(cached) = ctx.cache_get(sp, array_id) {
                ctx.log(format_args!("cache hit for {}", array));
                return Ok(cached);
            }
            ctx.cache_misses += 1;
        }

        let result = execute_array_step(array, ctx)?;

        if let Some(sp) = source_ptr {
            ctx.cache_put(
                sp,
                source_weak
                    .as_ref()
                    .vortex_expect("source_ptr implies source_weak"),
                &result,
            );
        }

        Ok(result)
    }
}

/// Inner single-step execution logic without caching.
fn execute_array_step(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    // 0. Check for canonical
    if let Some(canonical) = array.as_opt::<AnyCanonical>() {
        ctx.log(format_args!("-> canonical {}", array));
        return Ok(Canonical::from(canonical).into_array());
    }

    // 1. reduce (metadata-only rewrites)
    if let Some(reduced) = array.vtable().reduce(&array)? {
        ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
        reduced.statistics().inherit_from(array.statistics());
        return Ok(reduced);
    }

    // 2. reduce_parent (child-driven metadata-only rewrites)
    for child_idx in 0..array.nchildren() {
        let child = array.nth_child(child_idx).vortex_expect("checked length");
        if let Some(reduced_parent) = child.vtable().reduce_parent(&child, &array, child_idx)? {
            ctx.log(format_args!(
                "reduce_parent: child[{}]({}) rewrote {} -> {}",
                child_idx,
                child.encoding_id(),
                array,
                reduced_parent
            ));
            reduced_parent.statistics().inherit_from(array.statistics());
            return Ok(reduced_parent);
        }
    }

    // 3. execute_parent (child-driven optimized execution)
    for child_idx in 0..array.nchildren() {
        let child = array.nth_child(child_idx).vortex_expect("checked length");
        if let Some(executed_parent) = child
            .vtable()
            .execute_parent(&child, &array, child_idx, ctx)?
        {
            ctx.log(format_args!(
                "execute_parent: child[{}]({}) rewrote {} -> {}",
                child_idx,
                child.encoding_id(),
                array,
                executed_parent
            ));
            executed_parent
                .statistics()
                .inherit_from(array.statistics());
            return Ok(executed_parent);
        }
    }

    // 4. execute (optimized execution)
    ctx.log(format_args!("executing {}", array));
    let array = array
        .vtable()
        .execute(&array, ctx)
        .map(|c| c.into_array())?;
    array.statistics().inherit_from(array.statistics());
    ctx.log(format_args!("-> {}", array.as_ref()));

    Ok(array)
}

/// Extension trait for creating an execution context from a session.
pub trait VortexSessionExecute {
    /// Create a new execution context from this session.
    fn create_execution_ctx(&self) -> ExecutionCtx;
}

impl VortexSessionExecute for VortexSession {
    fn create_execution_ctx(&self) -> ExecutionCtx {
        ExecutionCtx::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::Columnar;
    use crate::Executable;
    use crate::IntoArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::executor::ExecutionCache;
    use crate::executor::ExecutionCtx;
    use crate::executor::ptr_id;
    use crate::expr::and;
    use crate::expr::gt;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::root;

    #[test]
    fn ptr_id_same_arc_is_equal() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let clone = array.clone();
        assert_eq!(ptr_id(&array), ptr_id(&clone));
    }

    #[test]
    fn ptr_id_different_arcs_are_not_equal() {
        let a = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let b = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        assert_ne!(ptr_id(&a), ptr_id(&b));
    }

    #[test]
    fn cache_forward_and_inverse() {
        use std::sync::Arc;

        let source = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let intermediate = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();
        let final_result = PrimitiveArray::from_iter([7i32, 8, 9]).into_array();

        let mut cache = ExecutionCache::new();
        let source_ptr = ptr_id(&source);
        let source_weak = Arc::downgrade(&source);

        // Store source → intermediate
        cache
            .forward
            .insert(source_ptr, (source_weak.clone(), intermediate.clone()));
        cache.inverse.insert(ptr_id(&intermediate), source_ptr);

        // Lookup by source pointer finds intermediate
        let (_, cached) = cache.forward.get(&source_ptr).vortex_expect("present");
        assert_eq!(ptr_id(cached), ptr_id(&intermediate));

        // Inverse map traces intermediate back to source
        assert_eq!(
            *cache
                .inverse
                .get(&ptr_id(&intermediate))
                .vortex_expect("present"),
            source_ptr,
        );

        // Update to final result
        cache
            .forward
            .insert(source_ptr, (source_weak.clone(), final_result.clone()));
        cache.inverse.insert(ptr_id(&final_result), source_ptr);

        // Forward now returns final result
        let (_, cached) = cache.forward.get(&source_ptr).vortex_expect("present");
        assert_eq!(ptr_id(cached), ptr_id(&final_result));

        // After source is dropped, the weak ref detects staleness
        drop(source);
        let (weak, _) = cache.forward.get(&source_ptr).vortex_expect("present");
        assert_eq!(weak.strong_count(), 0);
    }

    /// Tests that when `a > 5 && a < 10` is applied and executed, the shared reference to `a`
    /// produces cache hits.
    ///
    /// `apply()` builds a tree where `root()` resolves to the same `Arc` pointer in both the
    /// `gt` and `lt` sub-expressions. We use a dictionary-encoded array so that `a` requires
    /// execution (it is not already canonical). The first encounter populates the cache and
    /// the second encounter hits it.
    #[test]
    fn shared_subexpression_cache_hit() -> VortexResult<()> {
        // Wrap in MaskedArray so `a` is non-canonical and requires execution.
        let inner = PrimitiveArray::from_iter([1i32, 6, 7, 11]).into_array();
        let a = MaskedArray::try_new(inner, crate::validity::Validity::AllValid)?.into_array();

        // Build expression: a > 5 AND a < 10
        // apply() turns each root() into a.to_array(), so both gt and lt share the same Arc.
        let expr = and(gt(root(), lit(5i32)), lt(root(), lit(10i32)));
        let expr_array = a.apply(&expr)?;

        let mut ctx = ExecutionCtx::new(crate::LEGACY_SESSION.clone());
        let _result = Columnar::execute(expr_array, &mut ctx)?;

        assert!(
            ctx.cache_hits() > 0,
            "Expected cache hits for shared sub-expression `a`, got 0 \
             (hits={}, misses={}, size={})",
            ctx.cache_hits(),
            ctx.cache_misses(),
            ctx.cache_size(),
        );

        Ok(())
    }
}
