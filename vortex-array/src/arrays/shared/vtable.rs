// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::shared::SharedData;
use crate::arrays::shared::array::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;

vtable!(Shared, Shared, SharedData);

// TODO(ngates): consider hooking Shared into the iterative execution model. Cache either the
//  most executed, or after each iteration, and return a shared cache for each execution.
#[derive(Clone, Debug)]
pub struct Shared;

impl Shared {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.shared");
}

impl VTable for Shared {
    type ArrayData = SharedData;
    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &SharedData) -> &Self {
        &Shared
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &SharedData) -> usize {
        array.current_array_ref().len()
    }

    fn dtype(array: &SharedData) -> &DType {
        &array.dtype
    }

    fn stats(array: &SharedData) -> &ArrayStats {
        &array.stats
    }

    fn array_hash<H: std::hash::Hasher>(array: &SharedData, state: &mut H, precision: Precision) {
        let current = array.current_array_ref();
        current.array_hash(state, precision);
    }

    fn array_eq(array: &SharedData, other: &SharedData, precision: Precision) -> bool {
        let current = array.current_array_ref();
        let other_current = other.current_array_ref();
        current.array_eq(other_current, precision)
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

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_error::vortex_ensure!(
            slots.len() == 1,
            "SharedArray expects exactly 1 slot, got {}",
            slots.len()
        );
        let slot = slots
            .into_iter()
            .next()
            .vortex_expect("slots length already validated");
        array.set_source(slot);
        Ok(())
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn crate::serde::ArrayChildren,
    ) -> VortexResult<SharedData> {
        let child = children.get(0, dtype, len)?;
        Ok(SharedData::new(child))
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
