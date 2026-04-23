// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder kernels: drive decode directly into a canonical [`ArrayBuilder`].
//!
//! Where [`crate::executor`] runs an encoding toward a canonical [`ArrayRef`] stage by stage,
//! a builder kernel lets an encoding append its logical values straight into a builder that
//! the executor supplies. This avoids materializing intermediate canonical arrays when the
//! final destination already is a builder (e.g. `ChunkedArray` appending each chunk into a
//! single output builder).
//!
//! Encoding authors implement [`AppendToBuilderKernel`] and register it in
//! [`BuilderKernelSession`]. The executor dispatches through the type-erased
//! [`DynAppendToBuilderKernel`].

use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_session::registry::Id;
use vortex_session::registry::Registry;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::builders::ArrayBuilder;
use crate::executor::DonePredicate;
use crate::matcher::Matcher;

/// A step returned by a builder kernel describing what the executor should do next.
///
/// The kernel has already appended zero or more items into the owned builder it returns in the
/// accompanying [`BuilderResult`]. This enum tells the scheduler whether more work is needed.
pub enum BuilderStep {
    /// Kernel has finished appending this array; the scheduler may `finish()` the builder or
    /// continue driving an outer kernel.
    Done,

    /// The scheduler should execute the child at this slot index until the [`DonePredicate`]
    /// matches. When done, the scheduler extends the builder from the child, nulls the slot,
    /// and re-enters this kernel.
    ExecuteSlot(usize, DonePredicate),
}

impl Debug for BuilderStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuilderStep::Done => write!(f, "Done"),
            BuilderStep::ExecuteSlot(i, _) => f.debug_tuple("ExecuteSlot").field(i).finish(),
        }
    }
}

/// The result of a single step of a builder kernel.
///
/// `array` is the (possibly physically rewritten) array whose children the scheduler should
/// inspect when handling `step`. `builder` is the owned builder that is threaded through the
/// execution: the kernel may have appended into it already.
pub struct BuilderResult {
    array: ArrayRef,
    builder: Box<dyn ArrayBuilder>,
    step: BuilderStep,
}

impl BuilderResult {
    /// Decompose this result into its parts.
    pub fn into_parts(self) -> (ArrayRef, Box<dyn ArrayBuilder>, BuilderStep) {
        (self.array, self.builder, self.step)
    }
}

impl Debug for BuilderResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuilderResult")
            .field("array", &self.array)
            .field("step", &self.step)
            .finish()
    }
}

/// A typed kernel that appends directly into a canonical builder.
///
/// Encodings implement this trait to avoid materializing intermediate canonical arrays when
/// decoding. Registered per-encoding in [`BuilderKernelSession`].
///
/// The kernel receives a borrowed [`ArrayView`] into the `ArrayRef` that the executor owns;
/// the executor retains ownership of the array so that any follow-up `take_slot_unchecked`
/// calls see a unique `Arc` and can actually leave the taken slot as `None`.
pub trait AppendToBuilderKernel<V: VTable>: Debug + Send + Sync + 'static {
    /// Take one step of the builder-driven execution of `array`.
    ///
    /// The kernel owns `builder` for the duration of the step and returns it together with a
    /// [`BuilderStep`] describing what the executor should do next.
    fn append_to_builder(
        &self,
        array: ArrayView<'_, V>,
        builder: Box<dyn ArrayBuilder>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Box<dyn ArrayBuilder>, BuilderStep)>;
}

/// Type-erased version of [`AppendToBuilderKernel`] used for dynamic dispatch from the executor.
///
/// The array is passed by value so that the adapter can both downcast to the typed view and
/// return the same `ArrayRef` in [`BuilderResult`] without cloning — this keeps the `Arc`
/// refcount at one, which the executor relies on to actually mutate slots in place.
pub trait DynAppendToBuilderKernel: Send + Sync + Debug {
    fn append_to_builder(
        &self,
        array: ArrayRef,
        builder: Box<dyn ArrayBuilder>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BuilderResult>;
}

/// Bridges a concrete [`AppendToBuilderKernel<V>`] into the type-erased form.
pub struct AppendToBuilderKernelAdapter<V: VTable, K: AppendToBuilderKernel<V>> {
    kernel: K,
    _phantom: PhantomData<fn() -> V>,
}

impl<V: VTable, K: AppendToBuilderKernel<V>> AppendToBuilderKernelAdapter<V, K> {
    pub const fn new(kernel: K) -> Self {
        Self {
            kernel,
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable, K: AppendToBuilderKernel<V>> Debug for AppendToBuilderKernelAdapter<V, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppendToBuilderKernelAdapter")
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl<V: VTable, K: AppendToBuilderKernel<V>> DynAppendToBuilderKernel
    for AppendToBuilderKernelAdapter<V, K>
{
    fn append_to_builder(
        &self,
        array: ArrayRef,
        builder: Box<dyn ArrayBuilder>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BuilderResult> {
        let (builder, step) = {
            let view = V::try_match(&array).unwrap_or_else(|| {
                vortex_error::vortex_panic!(
                    "DynAppendToBuilderKernel received array with encoding that does not match the kernel's VTable"
                )
            });
            self.kernel.append_to_builder(view, builder, ctx)?
        };
        Ok(BuilderResult {
            array,
            builder,
            step,
        })
    }
}

/// Session variable holding the set of registered builder kernels, keyed by encoding ID.
///
/// Registration is typically performed alongside `ArrayPlugin` registration. The [`Default`]
/// implementation (see `crate::session`) pre-registers the built-in encodings so a freshly
/// constructed session still benefits from the builder-kernel fast path.
#[derive(Debug, Clone)]
pub struct BuilderKernelSession {
    kernels: Registry<std::sync::Arc<dyn DynAppendToBuilderKernel>>,
}

impl BuilderKernelSession {
    /// Returns an empty session with no registered kernels.
    pub fn empty() -> Self {
        Self {
            kernels: Registry::default(),
        }
    }

    /// Register a kernel for a specific encoding.
    pub fn register<V, K>(&self, id: impl Into<Id>, kernel: K)
    where
        V: VTable,
        K: AppendToBuilderKernel<V>,
    {
        let adapter: std::sync::Arc<dyn DynAppendToBuilderKernel> =
            std::sync::Arc::new(AppendToBuilderKernelAdapter::<V, K>::new(kernel));
        self.kernels.register(id, adapter);
    }

    /// Look up a registered kernel by encoding ID.
    pub fn find(&self, id: &Id) -> Option<std::sync::Arc<dyn DynAppendToBuilderKernel>> {
        self.kernels.find(id)
    }
}
