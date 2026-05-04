// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::reversed::ReversedArrayExt as _;
use crate::arrays::reversed::array::{CHILD_SLOT, SLOT_NAMES};
use crate::arrays::reversed::execute::reverse_canonical;
use crate::arrays::reversed::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Reversed`]-encoded Vortex array.
///
/// A lazy wrapper that yields the elements of the inner array in reverse order.
/// The reversal is applied at execution time via [`reverse_canonical`].
///
/// Use [`ArrayRef::reverse`] to construct one; the optimizer is applied immediately
/// and may eliminate the wrapper for well-known encodings.
pub type ReversedArray = Array<Reversed>;

/// Encoding tag for [`ReversedArray`].
#[derive(Clone, Debug)]
pub struct Reversed;

impl VTable for Reversed {
    type ArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.reversed");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots[CHILD_SLOT].is_some(),
            "ReversedArray child slot must be present"
        );
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "ReversedArray dtype {} does not match child dtype {}",
            dtype,
            child.dtype(),
        );
        vortex_ensure!(
            child.len() == len,
            "ReversedArray length {} does not match child length {}",
            len,
            child.len(),
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ReversedArray has no buffers (index {idx})")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES
            .get(idx)
            .copied()
            .unwrap_or_else(|| vortex_panic!("ReversedArray slot index {idx} out of bounds"))
            .to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("ReversedArray is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_bail!("ReversedArray is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Ensure the child is in canonical form before reversing.
        let array = require_child!(array, array.child(), CHILD_SLOT => AnyCanonical);
        debug_assert!(array.child().is_canonical());
        reverse_canonical(array.child(), ctx).map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<Reversed> for Reversed {
    fn scalar_at(
        array: ArrayView<'_, Reversed>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let reversed_index = array.len() - 1 - index;
        array.child().execute_scalar(reversed_index, ctx)
    }
}

impl ValidityVTable<Reversed> for Reversed {
    fn validity(array: ArrayView<'_, Reversed>) -> VortexResult<Validity> {
        let inner = array.child().validity()?;
        match inner {
            Validity::NonNullable => Ok(Validity::NonNullable),
            Validity::AllValid => Ok(Validity::AllValid),
            Validity::AllInvalid => Ok(Validity::AllInvalid),
            Validity::Array(arr) => Ok(Validity::Array(arr.reverse()?)),
        }
    }
}
