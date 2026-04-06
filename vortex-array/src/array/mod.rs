// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use arcref::ArcRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ExecutionCtx;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;
use crate::validity::Validity;

mod erased;
pub use erased::*;

mod typed;
pub use typed::*;

pub mod vtable;
pub use vtable::*;

mod view;
pub use view::*;

/// The public API trait for all Vortex arrays.
///
/// This trait is sealed and cannot be implemented outside of `vortex-array`.
/// Use [`ArrayRef`] as the primary handle for working with arrays.
#[doc(hidden)]
pub(crate) trait DynArray: 'static + private::Sealed + Send + Sync + Debug {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns the logical Vortex [`DType`] of the array.
    fn dtype(&self) -> &DType;

    /// Returns the vtable of the array.
    fn vtable(&self) -> &dyn DynVTable;

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

    /// Returns the slots of the array.
    fn slots(&self, this: &ArrayRef) -> Vec<Option<ArrayRef>>;

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
    fn dyn_array_eq(&self, other: &dyn Any, precision: crate::Precision) -> bool;
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

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn vtable(&self) -> &dyn DynVTable {
        &self.vtable
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

    fn slots(&self, this: &ArrayRef) -> Vec<Option<ArrayRef>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::slots(view).to_vec()
    }

    fn slot_name(&self, this: &ArrayRef, idx: usize) -> String {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::slot_name(view, idx)
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

    fn metadata(&self, this: &ArrayRef) -> VortexResult<Option<Vec<u8>>> {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        V::serialize(V::metadata(view)?)
    }

    fn metadata_fmt(&self, this: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result {
        let view = unsafe { ArrayView::new_unchecked(this, &self.data) };
        match V::metadata(view) {
            Err(e) => write!(f, "<serde error: {e}>"),
            Ok(metadata) => Debug::fmt(&metadata, f),
        }
    }

    fn dyn_array_hash(&self, state: &mut dyn Hasher, precision: crate::Precision) {
        let mut wrapper = HasherWrapper(state);
        self.len.hash(&mut wrapper);
        self.dtype.hash(&mut wrapper);
        self.vtable.id().hash(&mut wrapper);
        V::array_hash(&self.data, &mut wrapper, precision);
    }

    fn dyn_array_eq(&self, other: &dyn Any, precision: crate::Precision) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| {
            self.len == other.len
                && self.dtype == other.dtype
                && self.vtable.id() == other.vtable.id()
                && V::array_eq(&self.data, &other.data, precision)
        })
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
