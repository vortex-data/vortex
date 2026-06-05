// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
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
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::vtable::EmptyArrayData;
use crate::arrays::compaction::CompactionArrayExt;
use crate::arrays::compaction::array::CHILD_SLOT;
use crate::arrays::compaction::array::SLOT_NAMES;
use crate::arrays::compaction::compact::compact_canonical;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Compaction`]-encoded Vortex array.
///
/// See the [module docs](super) for semantics.
pub type CompactionArray = Array<Compaction>;

/// An encoding that normalizes its child into a compact canonical form when executed.
#[derive(Clone, Debug)]
pub struct Compaction;

impl VTable for Compaction {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.compaction");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots[CHILD_SLOT].is_some(),
            "CompactionArray child slot must be present"
        );
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "CompactionArray dtype {} does not match child dtype {}",
            dtype,
            child.dtype()
        );
        vortex_ensure!(
            child.len() == len,
            "CompactionArray length {} does not match child length {}",
            len,
            child.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("CompactionArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Compaction array is not serializable")
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
        vortex_bail!("Compaction array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Child encodings that can compact themselves more cheaply than a full decode (e.g.
        // `Dict` via garbage collection) intercept this before we get here, via their
        // `execute_parent` kernels. Everything else is decoded to canonical and compacted
        // structurally.
        let array = require_child!(array, array.child(), CHILD_SLOT => AnyCanonical);
        let canonical = array.child().as_::<AnyCanonical>().into();
        compact_canonical(canonical, ctx).map(ExecutionResult::done)
    }
}

impl OperationsVTable<Compaction> for Compaction {
    fn scalar_at(
        array: ArrayView<'_, Compaction>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child().execute_scalar(index, ctx)
    }
}

impl ValidityVTable<Compaction> for Compaction {
    fn validity(array: ArrayView<'_, Compaction>) -> VortexResult<Validity> {
        array.child().validity()
    }
}
