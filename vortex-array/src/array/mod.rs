// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::Id;

use crate::ExecutionCtx;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionResult;
use crate::executor::ExecutionStep;
use crate::scalar::Scalar;
use crate::validity::Validity;

mod erased;
pub use erased::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub(crate) use foreign::*;

mod typed;
pub use typed::*;

pub mod vtable;
pub use vtable::*;

mod view;
use smallvec::SmallVec;
pub use view::*;

use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

/// The slots of an array: a collection of optional child arrays.
///
/// Most encodings have 4 or fewer slots, so we use a `SmallVec` to avoid
/// heap allocation in the common case.
pub type ArraySlots = SmallVec<[Option<ArrayRef>; 4]>;

/// The public API trait for all Vortex arrays.
///
/// This trait is sealed and cannot be implemented outside of `vortex-array`.
/// Use [`ArrayRef`] as the primary handle for working with arrays.
#[doc(hidden)]
pub(crate) trait DynArrayData: 'static + private::Sealed + Send + Sync + Debug {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the array as a mutable reference to a generic [`Any`] trait object.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Returns the [`Validity`] of the array.
    fn validity(&self, this: &ArrayRef) -> VortexResult<Validity>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(
        &self,
        this: &ArrayRef,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    // --- Visitor methods (formerly in ArrayVisitor) ---

    /// Returns the children of the array.
    fn children(&self, this: &ArrayRef) -> Vec<ArrayRef>;

    /// Returns the number of children of the array.
    fn nchildren(&self, this: &ArrayRef) -> usize;

    /// Returns the nth child of the array without allocating a Vec.
    ///
    /// Returns `None` if the index is out of bounds.
    fn nth_child(&self, this: &ArrayRef, idx: usize) -> Option<ArrayRef>;

    /// Returns the names of the children of the array.
    fn children_names(&self, this: &ArrayRef) -> Vec<String>;

    /// Returns the array's children with their names.
    fn named_children(&self, this: &ArrayRef) -> Vec<(String, ArrayRef)>;

    /// Returns the buffers of the array.
    fn buffers(&self, this: &ArrayRef) -> Vec<ByteBuffer>;

    /// Returns the buffer handles of the array.
    fn buffer_handles(&self, this: &ArrayRef) -> Vec<BufferHandle>;

    /// Returns the names of the buffers of the array.
    fn buffer_names(&self, this: &ArrayRef) -> Vec<String>;

    /// Returns the array's buffers with their names.
    fn named_buffers(&self, this: &ArrayRef) -> Vec<(String, BufferHandle)>;

    /// Returns the number of buffers of the array.
    fn nbuffers(&self, this: &ArrayRef) -> usize;

    /// Returns the name of the slot at the given index.
    fn slot_name(&self, this: &ArrayRef, idx: usize) -> String;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;

    /// Hashes the array contents including len, dtype, and encoding id.
    fn dyn_array_hash(&self, state: &mut dyn Hasher, accuracy: crate::EqMode);

    /// Compares two arrays of the same concrete type for equality.
    fn dyn_array_eq(&self, other: &ArrayRef, accuracy: crate::EqMode) -> bool;

    /// Returns a new array with the given slots.
    fn with_slots(&self, this: &ArrayRef, slots: ArraySlots) -> VortexResult<ArrayRef>;

    /// Returns a new array with the given slots, bypassing encoding-level validation.
    ///
    /// Used by the executor to temporarily carry an array that has had one of its child slots
    /// taken out (leaving `None`) without panicking `V::validate`. The caller must ensure the
    /// missing slot is filled back in (via `put_slot_unchecked`) or driven to completion by the
    /// builder path before the array becomes externally observable.
    ///
    /// # Safety
    ///
    /// The array returned may have slots whose content does not match the encoding's normal
    /// invariants. Callers must re-establish those invariants before handing the array to
    /// anything outside the executor.
    unsafe fn with_slots_unchecked(&self, this: &ArrayRef, slots: ArraySlots) -> ArrayRef;

    /// Attempt to reduce the array to a simpler representation.
    fn reduce(&self, this: &ArrayRef) -> VortexResult<Option<ArrayRef>>;

    /// Attempt to reduce the parent of this array.
    fn reduce_parent(
        &self,
        this: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    /// Execute the array by taking a single encoding-specific execution step.
    ///
    /// This is the checked entry point. If the encoding reports
    /// [`ExecutionStep::Done`](crate::ExecutionStep::Done), implementations must validate that the
    /// returned array preserves this array's logical `len` and `dtype`, and must transfer this
    /// array's statistics to the returned array.
    fn execute(&self, this: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

    /// Execute the array by taking a single encoding-specific execution step without applying
    /// `Done`-result postconditions.
    ///
    /// This exists for the iterative executor, which may call into `execute` on suspended
    /// executor-private arrays whose slots temporarily contain `None`. In that mode the executor
    /// itself is responsible for deciding when a `Done` result represents a real logical array,
    /// enforcing any `len`/`dtype` invariants, and transferring statistics.
    ///
    /// # Safety
    /// The `array` returned should have it's `DType` and len checked
    /// (optionally it should have its stats propagated from `this`).
    unsafe fn execute_unchecked(
        &self,
        this: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult>;

    /// Attempt to execute the parent of this array.
    fn execute_parent(
        &self,
        this: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;

    /// Execute the scalar at the given index.
    ///
    /// This method panics if the index is out of bounds for the array.
    fn execute_scalar(
        &self,
        this: &ArrayRef,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar>;
}

/// Trait for converting a type into a Vortex [`ArrayRef`].
pub trait IntoArray {
    fn into_array(self) -> ArrayRef;
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for ArrayData<V> {}
}

// =============================================================================
// New path: DynArrayData and supporting trait impls for ArrayData<V>
// =============================================================================

/// DynArrayData implementation for [`ArrayData<V>`].
///
/// This is self-contained: identity methods use `ArrayData<V>`'s own fields (dtype, len, stats),
/// while data-access methods delegate to VTable methods on the inner `V::TypedArrayData`.
impl<V: VTable> DynArrayData for ArrayData<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn validity(&self, this: &ArrayRef) -> VortexResult<Validity> {
        if this.dtype().is_nullable() {
            let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
            let validity = <V::ValidityVTable as ValidityVTable<V>>::validity(view)?;
            if let Validity::Array(array) = &validity {
                vortex_ensure!(array.len() == this.len(), "Validity array length mismatch");
                vortex_ensure!(
                    matches!(array.dtype(), DType::Bool(Nullability::NonNullable)),
                    "Validity array is not non-nullable boolean: {}",
                    this.encoding_id(),
                );
            }
            Ok(validity)
        } else {
            Ok(Validity::NonNullable)
        }
    }

    fn append_to_builder(
        &self,
        this: &ArrayRef,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if builder.dtype() != this.dtype() {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                this.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::append_to_builder(view, builder, ctx)?;

        assert_eq!(
            len + this.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            this.encoding_id(),
        );
        Ok(())
    }

    fn children(&self, this: &ArrayRef) -> Vec<ArrayRef> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nchildren(view)).map(|i| V::child(view, i)).collect()
    }

    fn nchildren(&self, this: &ArrayRef) -> usize {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::nchildren(view)
    }

    fn nth_child(&self, this: &ArrayRef, idx: usize) -> Option<ArrayRef> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (idx < V::nchildren(view)).then(|| V::child(view, idx))
    }

    fn children_names(&self, this: &ArrayRef) -> Vec<String> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nchildren(view))
            .map(|i| V::child_name(view, i))
            .collect()
    }

    fn named_children(&self, this: &ArrayRef) -> Vec<(String, ArrayRef)> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nchildren(view))
            .map(|i| (V::child_name(view, i), V::child(view, i)))
            .collect()
    }

    fn buffers(&self, this: &ArrayRef) -> Vec<ByteBuffer> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nbuffers(view))
            .map(|i| V::buffer(view, i).to_host_sync())
            .collect()
    }

    fn buffer_handles(&self, this: &ArrayRef) -> Vec<BufferHandle> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nbuffers(view)).map(|i| V::buffer(view, i)).collect()
    }

    fn buffer_names(&self, this: &ArrayRef) -> Vec<String> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nbuffers(view))
            .filter_map(|i| V::buffer_name(view, i))
            .collect()
    }

    fn named_buffers(&self, this: &ArrayRef) -> Vec<(String, BufferHandle)> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        (0..V::nbuffers(view))
            .filter_map(|i| V::buffer_name(view, i).map(|name| (name, V::buffer(view, i))))
            .collect()
    }

    fn nbuffers(&self, this: &ArrayRef) -> usize {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::nbuffers(view)
    }

    fn slot_name(&self, this: &ArrayRef, idx: usize) -> String {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::slot_name(view, idx)
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.data, f)
    }

    fn dyn_array_hash(&self, state: &mut dyn Hasher, accuracy: crate::EqMode) {
        let mut wrapper = HasherWrapper(state);
        // Note: metadata (len, dtype, encoding_id) and slots are hashed by ArrayRef.
        self.data.array_hash(&mut wrapper, accuracy);
    }

    fn dyn_array_eq(&self, other: &ArrayRef, accuracy: crate::EqMode) -> bool {
        // Note: metadata (len, dtype, encoding_id) and slots are compared by ArrayRef.
        other
            .dyn_array()
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other_inner| self.data.array_eq(&other_inner.data, accuracy))
    }

    fn with_slots(&self, this: &ArrayRef, slots: ArraySlots) -> VortexResult<ArrayRef> {
        let stats = this.statistics().to_owned();
        Ok(Array::<V>::try_from_parts(
            ArrayParts::new(
                self.vtable.clone(),
                this.dtype().clone(),
                this.len(),
                self.data.clone(),
            )
            .with_slots(slots),
        )?
        .with_stats_set(stats)
        .into_array())
    }

    unsafe fn with_slots_unchecked(&self, this: &ArrayRef, slots: ArraySlots) -> ArrayRef {
        // SAFETY: we intentionally skip `V::validate` here. Caller guarantees that the resulting
        // array is either repaired or not externally observed.
        let store = unsafe {
            ArrayInner::<ArrayData<V>>::new_unchecked(
                self.vtable.clone(),
                this.len(),
                this.dtype().clone(),
                self.data.clone(),
                slots,
                this.statistics().to_array_stats(),
            )
        };
        ArrayRef::from_inner(Arc::new(store))
    }

    fn reduce(&self, this: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        let Some(reduced) = V::reduce(view)? else {
            return Ok(None);
        };
        vortex_ensure!(
            reduced.len() == this.len(),
            "Reduced array length mismatch from {} to {}",
            this.encoding_id(),
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == this.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            this.encoding_id(),
            reduced.encoding_id()
        );
        Ok(Some(reduced))
    }

    fn reduce_parent(
        &self,
        this: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        let Some(reduced) = V::reduce_parent(view, parent, child_idx)? else {
            return Ok(None);
        };

        vortex_ensure!(
            reduced.len() == parent.len(),
            "Reduced array length mismatch from {} to {}",
            parent.encoding_id(),
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == parent.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            parent.encoding_id(),
            reduced.encoding_id()
        );

        Ok(Some(reduced))
    }

    fn execute(&self, this: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let len = this.len();
        let dtype = this.dtype().clone();
        let stats = this.statistics().to_array_stats();
        let result = unsafe { self.execute_unchecked(this, ctx)? };

        if matches!(result.step(), ExecutionStep::Done) {
            if cfg!(debug_assertions) {
                vortex_ensure!(
                    result.array().len() == len,
                    "Result length mismatch for {:?}",
                    self.vtable
                );
                vortex_ensure!(
                    result.array().dtype() == &dtype,
                    "Executed canonical dtype mismatch for {:?}",
                    self.vtable
                );
            }

            result
                .array()
                .statistics()
                .set_iter(crate::stats::StatsSet::from(stats).into_iter());
        }

        Ok(result)
    }

    unsafe fn execute_unchecked(
        &self,
        this: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult> {
        let typed = Array::<V>::try_from_array_ref(this)
            .map_err(|_| vortex_err!("Failed to downcast array for execute"))
            .vortex_expect("Failed to downcast array for execute");
        V::execute(typed, ctx)
    }

    fn execute_parent(
        &self,
        this: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        let Some(result) = V::execute_parent(view, parent, child_idx, ctx)? else {
            return Ok(None);
        };

        if cfg!(debug_assertions) {
            vortex_ensure!(
                result.len() == parent.len(),
                "Executed parent canonical length mismatch"
            );
            vortex_ensure!(
                result.dtype() == parent.dtype(),
                "Executed parent canonical dtype mismatch"
            );
        }

        Ok(Some(result))
    }

    fn execute_scalar(
        &self,
        this: &ArrayRef,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        <V::OperationsVTable as OperationsVTable<V>>::scalar_at(view, index, ctx)
    }
}

/// Wrapper around `&mut dyn Hasher` that implements `Hasher` (and is `Sized`).
struct HasherWrapper<'a>(&'a mut dyn Hasher);

impl Hasher for HasherWrapper<'_> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

/// ArrayId is a globally unique name for the array's vtable.
pub type ArrayId = Id;
