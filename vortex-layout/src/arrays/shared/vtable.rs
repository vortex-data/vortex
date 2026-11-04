// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::OperationsVTable;
use vortex_array::ProbeState;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VTable;
use vortex_array::ValidityVTable;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::validity::Validity;
use vortex_array::with_empty_buffers;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::arrays::shared::SharedArrayExt;
use crate::arrays::shared::SharedData;
use crate::arrays::shared::SharedSlots;

/// A [`Shared`]-encoded Vortex array.
pub type SharedArray = Array<Shared>;

// TODO(ngates): consider hooking Shared into the iterative execution model. Cache either the
//  most executed, or after each iteration, and return a shared cache for each execution.
#[derive(Clone, Debug)]
pub struct Shared;

impl ArrayHash for SharedData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for SharedData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl VTable for Shared {
    type TypedArrayData = SharedData;
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
        let source = slots[SharedSlots::SOURCE]
            .as_ref()
            .vortex_expect("SharedArray source slot must be present");
        vortex_error::vortex_ensure_eq!(source.dtype(), dtype, "SharedArray dtype mismatch");
        vortex_error::vortex_ensure_eq!(source.len(), len, "SharedArray len mismatch");
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SharedSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],

        _buffers: &[BufferHandle],
        _children: &dyn vortex_array::serde::ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        array
            .get_or_compute(|source| source.clone().execute::<Canonical>(ctx))
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(comparison) = parent.as_opt::<ExactScalarFn<Binary>>() else {
            return Ok(None);
        };
        if !comparison.options.is_comparison() {
            return Ok(None);
        }
        // A metadata-only comparison rewrite can use encoded values without decoding the
        // shared dictionary. Other operations retain the shared canonicalization boundary.
        let source = array.current_array_ref();
        // SAFETY: The source and cached form have the same logical values as this wrapper.
        let parent = unsafe { parent.clone().with_slot(child_idx, source.clone())? };
        source.reduce_parent(&parent, child_idx)
    }
}

/// A probe over whichever array [`SharedArrayExt::current_array_ref`] resolves to. The source
/// can be replaced by the computed result between reads, so the retained probe is rebuilt when
/// the array it was built over is no longer the current one.
#[derive(Default)]
pub struct SharedProbeState {
    current: Option<RepeatedArrayProbe>,
}

impl OperationsVTable<Shared> for Shared {
    type ProbeState = SharedProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Shared>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let current = array.current_array_ref();
        let Some(retained) = state.retained() else {
            return current.probe().execute_scalar(index, ctx);
        };
        let probe = match &mut retained.current {
            Some(probe) if ArrayRef::ptr_eq(probe.array(), current) => probe,
            slot => slot.insert(RepeatedArrayProbe::new(current.clone())),
        };
        probe.execute_scalar(index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Shared>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<Shared> for Shared {
    fn validity(array: ArrayView<'_, Shared>) -> VortexResult<Validity> {
        array.current_array_ref().validity()
    }
}
