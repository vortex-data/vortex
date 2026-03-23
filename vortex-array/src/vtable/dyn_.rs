// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayAdapter;
use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionResult;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::vtable::VTable;

/// ArrayId is a globally unique name for the array's vtable.
pub type ArrayId = ArcRef<str>;

/// Reference-counted DynVTable
pub type DynVTableRef = Arc<dyn DynVTable>;

/// Dynamically typed vtable trait.
///
/// This trait contains the implementation API for Vortex arrays, allowing us to keep the public
/// [`DynArray`] trait API to a minimum.
pub trait DynVTable: 'static + Send + Sync + Debug {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        id: ArrayId,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;
    fn with_children(&self, array: &ArrayRef, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;

    /// See [`VTable::reduce`]
    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;

    /// See [`VTable::reduce_parent`]
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    /// See [`VTable::execute`]
    fn execute(&self, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

    /// See [`VTable::execute_parent`]
    fn execute_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

impl<V: VTable> DynVTable for V {
    fn build(
        &self,
        _id: ArrayId,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let metadata = V::deserialize(metadata, dtype, len, buffers, session)?;
        let array = V::build(dtype, len, &metadata, buffers, children)?;
        assert_eq!(array.len(), len, "Array length mismatch after building");
        assert_eq!(array.dtype(), dtype, "Array dtype mismatch after building");
        Ok(array.into_array())
    }

    fn with_children(&self, array: &ArrayRef, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut array = array.as_::<V>().clone();
        V::with_children(&mut array, children)?;
        Ok(array.into_array())
    }

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce(downcast::<V>(array))? else {
            return Ok(None);
        };
        vortex_ensure!(
            reduced.len() == array.len(),
            "Reduced array length mismatch from {} to {}",
            array.encoding_id(),
            reduced.encoding_id()
        );
        vortex_ensure!(
            reduced.dtype() == array.dtype(),
            "Reduced array dtype mismatch from {} to {}",
            array.encoding_id(),
            reduced.encoding_id()
        );
        Ok(Some(reduced))
    }

    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce_parent(downcast::<V>(array), parent, child_idx)? else {
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

    fn execute(&self, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Capture metadata before the move for post-validation and stats inheritance.
        let len = array.len();
        let dtype = array.dtype().clone();
        let stats = array.statistics().to_owned();

        let owned = downcast_owned::<V>(array);
        let result = V::execute(owned, ctx)?;

        if matches!(result.step(), ExecutionStep::Done) {
            if cfg!(debug_assertions) {
                vortex_ensure!(
                    result.array().len() == len,
                    "Result length mismatch for {:?}",
                    self
                );
                vortex_ensure!(
                    result.array().dtype() == &dtype,
                    "Executed canonical dtype mismatch for {:?}",
                    self
                );
            }

            // TODO(ngates): do we want to do this on every execution? We used to in to_canonical.
            result.array().statistics().set_iter(stats.into_iter());
        }

        Ok(result)
    }

    fn execute_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(result) = V::execute_parent(downcast::<V>(array), parent, child_idx, ctx)? else {
            return Ok(None);
        };

        if cfg!(debug_assertions) {
            vortex_ensure!(
                result.as_ref().len() == parent.len(),
                "Executed parent canonical length mismatch"
            );
            vortex_ensure!(
                result.as_ref().dtype() == parent.dtype(),
                "Executed parent canonical dtype mismatch"
            );
        }

        Ok(Some(result))
    }
}

fn downcast<V: VTable>(array: &ArrayRef) -> &V::Array {
    array
        .as_any()
        .downcast_ref::<ArrayAdapter<V>>()
        .vortex_expect("Failed to downcast array to expected encoding type")
        .as_inner()
}

/// Downcast an `ArrayRef` into an `Arc<V::Array>` without cloning.
///
/// This is a zero-cost pointer cast leveraging the `#[repr(transparent)]` layout of
/// [`ArrayAdapter`].
fn downcast_owned<V: VTable>(array: ArrayRef) -> Arc<V::Array> {
    let adapter: Arc<ArrayAdapter<V>> = array
        .as_any_arc()
        .downcast::<ArrayAdapter<V>>()
        .ok()
        .vortex_expect("Failed to downcast array to expected encoding type");
    // SAFETY: ArrayAdapter<V> is #[repr(transparent)] over V::Array,
    // so Arc<ArrayAdapter<V>> and Arc<V::Array> have identical layout.
    let raw = Arc::into_raw(adapter) as *const V::Array;
    unsafe { Arc::from_raw(raw) }
}

/// Upcast an `Arc<V::Array>` into an `ArrayRef` without cloning.
///
/// This is a zero-cost pointer cast leveraging the `#[repr(transparent)]` layout of
/// [`ArrayAdapter`]. It is the reverse of `downcast_owned`.
pub(crate) fn upcast_array<V: VTable>(array: Arc<V::Array>) -> ArrayRef {
    // SAFETY: ArrayAdapter<V> is #[repr(transparent)] over V::Array,
    // so Arc<V::Array> and Arc<ArrayAdapter<V>> have identical layout.
    let raw = Arc::into_raw(array) as *const ArrayAdapter<V>;
    unsafe { Arc::from_raw(raw) }
}

impl<V: VTable> Debug for ArrayVTableAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Encoding<{}>", type_name::<V>())
    }
}

impl<V: VTable> From<V> for &'static dyn DynVTable {
    fn from(_vtable: V) -> Self {
        const { &ArrayVTableAdapter::<V>(PhantomData) }
    }
}

pub trait ArrayVTableExt {
    /// Wraps the vtable into an [`DynVTable`] by static reference.
    fn vtable() -> &'static dyn DynVTable;
}

impl<V: VTable> ArrayVTableExt for V {
    fn vtable() -> &'static dyn DynVTable {
        const { &ArrayVTableAdapter::<V>(PhantomData) }
    }
}

mod private {
    use super::ArrayVTableAdapter;
    use crate::vtable::VTable;

    pub trait Sealed {}
    impl<V: VTable> Sealed for ArrayVTableAdapter<V> {}
}
