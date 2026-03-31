// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionResult;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable::Array;
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
    /// Clone this vtable into a `Box<dyn DynVTable>`.
    fn clone_boxed(&self) -> Box<dyn DynVTable>;

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
    /// See [`VTable::with_slots`]
    fn with_slots(&self, array: ArrayRef, slots: Vec<Option<ArrayRef>>) -> VortexResult<ArrayRef>;

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
    fn clone_boxed(&self) -> Box<dyn DynVTable> {
        Box::new(self.clone())
    }

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
        let inner = V::build(dtype, len, &metadata, buffers, children)?;
        // Validate the inner array's properties before wrapping.
        assert_eq!(V::len(&inner), len, "Array length mismatch after building");
        assert_eq!(
            V::dtype(&inner),
            dtype,
            "Array dtype mismatch after building"
        );
        // Wrap in Array<V> for safe downcasting.
        // SAFETY: We just validated that V::len(&inner) == len and V::dtype(&inner) == dtype.
        let array = unsafe {
            Array::new_unchecked(
                self.clone(),
                dtype.clone(),
                len,
                inner,
                ArrayStats::default(),
            )
        };
        Ok(array.into_array())
    }

    fn with_slots(&self, array: ArrayRef, slots: Vec<Option<ArrayRef>>) -> VortexResult<ArrayRef> {
        let arc = downcast_owned::<V>(array);
        let mut inner = Arc::try_unwrap(arc).unwrap_or_else(|arc| arc.as_ref().clone());
        V::with_slots(&mut inner.array, slots)?;
        Ok(inner.into_array())
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

/// Borrow-downcast an `ArrayRef` to `&Array<V>`.
fn downcast<V: VTable>(array: &ArrayRef) -> &Array<V> {
    array
        .as_any()
        .downcast_ref::<Array<V>>()
        .vortex_expect("Failed to downcast array to expected encoding type")
}

/// Downcast an `ArrayRef` into an `Arc<Array<V>>`.
fn downcast_owned<V: VTable>(array: ArrayRef) -> Arc<Array<V>> {
    let any_arc = array.as_any_arc();
    any_arc
        .downcast::<Array<V>>()
        .ok()
        .vortex_expect("Failed to downcast array to expected encoding type")
}
