// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::TakeArray;
use crate::arrays::TakeVTable;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::vtable::VTable;

pub trait TakeReduce: VTable {
    /// Take elements from an array at the given indices without reading buffers.
    ///
    /// This trait is for take implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if taking requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(array: &Self::Array, indices: &dyn Array) -> VortexResult<Option<ArrayRef>>;
}

pub trait TakeExecute: VTable {
    /// Take elements from an array at the given indices, potentially reading buffers.
    ///
    /// Unlike [`TakeReduce`], this trait is for take implementations that may need to read
    /// and execute on the underlying buffers to produce the result.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(
        array: &Self::Array,
        indices: &dyn Array,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Common preconditions for take operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the take operation,
/// or `None` if the take should proceed normally.
pub fn precondition<V: VTable>(array: &V::Array, indices: &dyn Array) -> Option<ArrayRef> {
    // Fast-path for empty indices.
    if indices.is_empty() {
        return Some(Canonical::empty(array.dtype()).into_array());
    }

    None
}

#[derive(Default, Debug)]
pub struct TakeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeReduceAdaptor<V>
where
    V: TakeReduce,
{
    type Parent = TakeVTable;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: &TakeArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, parent.indices()) {
            return Ok(Some(result));
        }
        <V as TakeReduce>::take(array, parent.indices())
    }
}

#[derive(Default, Debug)]
pub struct TakeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeExecuteAdaptor<V>
where
    V: TakeExecute,
{
    type Parent = TakeVTable;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, parent.indices()) {
            return Ok(Some(result));
        }
        <V as TakeExecute>::take(array, parent.indices(), ctx)
    }
}
