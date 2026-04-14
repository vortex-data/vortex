// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::shared::SharedArrayExt;
use crate::arrays::shared::SharedData;
use crate::arrays::shared::array::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// A [`Shared`]-encoded Vortex array.
pub type SharedArray = Array<Shared>;

// TODO(ngates): consider hooking Shared into the iterative execution model. Cache either the
//  most executed, or after each iteration, and return a shared cache for each execution.
#[derive(Clone, Debug)]
pub struct Shared;

impl ArrayHash for SharedData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for SharedData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

impl VTable for Shared {
    type ArrayData = SharedData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.shared");
        *ID
    }

    fn validate(
        &self,
        _data: &SharedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let source = slots[0]
            .as_ref()
            .vortex_expect("SharedArray source slot must be present");
        vortex_error::vortex_ensure!(source.dtype() == dtype, "SharedArray dtype mismatch");
        vortex_error::vortex_ensure!(source.len() == len, "SharedArray len mismatch");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("SharedArray has no buffers")
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
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],

        _buffers: &[BufferHandle],
        _children: &dyn crate::serde::ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        array
            .get_or_compute(|source| source.clone().execute::<Canonical>(ctx))
            .map(ExecutionResult::done)
    }
}
impl OperationsVTable<Shared> for Shared {
    fn scalar_at(
        array: ArrayView<'_, Shared>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.current_array_ref().scalar_at(index)
    }
}

impl ValidityVTable<Shared> for Shared {
    fn validity(array: ArrayView<'_, Shared>) -> VortexResult<Validity> {
        array.current_array_ref().validity()
    }
}
