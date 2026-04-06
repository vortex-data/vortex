// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionResult;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayInner;
use crate::array::VTable;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;

/// Reference-counted DynVTable
pub type DynVTableRef = Arc<dyn DynVTable>;

/// Dynamically typed vtable trait.
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
        // Wrap in ArrayInner<V> for safe downcasting.
        // SAFETY: We just validated that V::len(&inner) == len and V::dtype(&inner) == dtype.
        let array = unsafe {
            ArrayInner::from_data_unchecked(
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
        let typed = array
            .as_opt::<V>()
            .vortex_expect("Failed to downcast array");
        let mut data = typed.data().clone();
        V::with_slots(&mut data, slots)?;
        Ok(Array::<V>::try_from_data(data)?.into_array())
    }

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(reduced) = V::reduce(array.as_::<V>())? else {
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
        let Some(reduced) = V::reduce_parent(array.as_::<V>(), parent, child_idx)? else {
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

        let typed = Array::<V>::try_from_array_ref(array)
            .map_err(|_| vortex_err!("Failed to downcast array for execute"))
            .vortex_expect("Failed to downcast array for execute");
        let result = V::execute(typed, ctx)?;

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
        let Some(result) = V::execute_parent(array.as_::<V>(), parent, child_idx, ctx)? else {
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
