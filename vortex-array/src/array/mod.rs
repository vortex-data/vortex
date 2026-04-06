// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use arcref::ArcRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ExecutionCtx;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionResult;
use crate::executor::ExecutionStep;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;
use crate::validity::Validity;

mod erased;
pub use erased::*;

mod plugin;
pub use plugin::*;

mod typed;
pub use typed::*;

pub mod vtable;
pub use vtable::*;

mod view;
pub use view::*;

use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

/// The public API trait for all Vortex arrays.
///
/// This trait is sealed and cannot be implemented outside of `vortex-array`.
/// Use [`ArrayRef`] as the primary handle for working with arrays.
#[doc(hidden)]
pub(crate) trait DynArray: 'static + private::Sealed + Send + Sync + Debug {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Converts an owned array allocation into an owned [`Any`] allocation for downcasting.
    fn into_any_arc(self: std::sync::Arc<Self>) -> std::sync::Arc<dyn Any + Send + Sync>;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns the logical Vortex [`DType`] of the array.
    fn dtype(&self) -> &DType;

    /// Returns the slots of the array.
    fn slots(&self) -> &[Option<ArrayRef>];

    /// Returns the encoding ID of the array.
    fn encoding_id(&self) -> ArrayId;

    /// Fetch the scalar at the given index.
    ///
    /// This method panics if the index is out of bounds for the array.
    fn scalar_at(&self, this: &ArrayRef, index: usize) -> VortexResult<Scalar>;

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

    /// Returns the statistics of the array.
    fn statistics(&self) -> &ArrayStats;

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

    /// Returns the serialized metadata of the array, or `None` if the array does not
    /// support serialization.
    fn metadata(&self, this: &ArrayRef) -> VortexResult<Option<Vec<u8>>>;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, this: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result;

    /// Hashes the array contents including len, dtype, and encoding id.
    fn dyn_array_hash(&self, state: &mut dyn Hasher, precision: crate::Precision);

    /// Compares two arrays of the same concrete type for equality.
    fn dyn_array_eq(&self, other: &ArrayRef, precision: crate::Precision) -> bool;

    /// Returns a new array with the given slots.
    fn with_slots(&self, this: ArrayRef, slots: Vec<Option<ArrayRef>>) -> VortexResult<ArrayRef>;

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
    fn execute(&self, this: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

    /// Attempt to execute the parent of this array.
    fn execute_parent(
        &self,
        this: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Trait for converting a type into a Vortex [`ArrayRef`].
pub trait IntoArray {
    fn into_array(self) -> ArrayRef;
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for ArrayInner<V> {}
}

// =============================================================================
// New path: DynArray and supporting trait impls for ArrayInner<V>
// =============================================================================

/// DynArray implementation for [`ArrayInner<V>`].
///
/// This is self-contained: identity methods use `ArrayInner<V>`'s own fields (dtype, len, stats),
/// while data-access methods delegate to VTable methods on the inner `V::ArrayData`.
impl<V: VTable> DynArray for ArrayInner<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any_arc(self: std::sync::Arc<Self>) -> std::sync::Arc<dyn Any + Send + Sync> {
        self
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn slots(&self) -> &[Option<ArrayRef>] {
        &self.slots
    }

    fn encoding_id(&self) -> ArrayId {
        self.vtable.id()
    }

    fn scalar_at(&self, this: &ArrayRef, index: usize) -> VortexResult<Scalar> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        <V::OperationsVTable as OperationsVTable<V>>::scalar_at(
            view,
            index,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
    }

    fn validity(&self, this: &ArrayRef) -> VortexResult<Validity> {
        if self.dtype.is_nullable() {
            let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
            let validity = <V::ValidityVTable as ValidityVTable<V>>::validity(view)?;
            if let Validity::Array(array) = &validity {
                vortex_ensure!(array.len() == self.len, "Validity array length mismatch");
                vortex_ensure!(
                    matches!(array.dtype(), DType::Bool(Nullability::NonNullable)),
                    "Validity array is not non-nullable boolean: {}",
                    self.vtable.id(),
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
        if builder.dtype() != &self.dtype {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype,
                builder.dtype(),
            );
        }
        let len = builder.len();

        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::append_to_builder(view, builder, ctx)?;

        assert_eq!(
            len + self.len,
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.vtable.id(),
        );
        Ok(())
    }

    fn statistics(&self) -> &ArrayStats {
        &self.stats
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

    fn metadata(&self, this: &ArrayRef) -> VortexResult<Option<Vec<u8>>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::serialize(view)
    }

    fn metadata_fmt(&self, this: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::fmt_metadata(view, f)
    }

    fn dyn_array_hash(&self, state: &mut dyn Hasher, precision: crate::Precision) {
        let mut wrapper = HasherWrapper(state);
        self.len.hash(&mut wrapper);
        self.dtype.hash(&mut wrapper);
        self.vtable.id().hash(&mut wrapper);
        self.slots.len().hash(&mut wrapper);
        for slot in &self.slots {
            slot.array_hash(&mut wrapper, precision);
        }
        self.data.array_hash(&mut wrapper, precision);
    }

    fn dyn_array_eq(&self, other: &ArrayRef, precision: crate::Precision) -> bool {
        other
            .inner()
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other_inner| {
                self.len == other.len()
                    && self.dtype == *other.dtype()
                    && self.vtable.id() == other.encoding_id()
                    && self.slots.len() == other_inner.slots.len()
                    && self
                        .slots
                        .iter()
                        .zip(other_inner.slots.iter())
                        .all(|(slot, other_slot)| slot.array_eq(other_slot, precision))
                    && self.data.array_eq(&other_inner.data, precision)
            })
    }

    fn with_slots(&self, this: ArrayRef, slots: Vec<Option<ArrayRef>>) -> VortexResult<ArrayRef> {
        let data = self.data.clone();
        let stats = this.statistics().to_owned();
        Ok(Array::<V>::try_from_parts(
            ArrayParts::new(self.vtable.clone(), this.dtype().clone(), this.len(), data)
                .with_slots(slots),
        )?
        .with_stats_set(stats)
        .into_array())
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
        let stats = this.statistics().to_owned();

        let typed = Array::<V>::try_from_array_ref(this)
            .map_err(|_| vortex_err!("Failed to downcast array for execute"))
            .vortex_expect("Failed to downcast array for execute");
        let result = V::execute(typed, ctx)?;

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

            result.array().statistics().set_iter(stats.into_iter());
        }

        Ok(result)
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
pub type ArrayId = ArcRef<str>;
