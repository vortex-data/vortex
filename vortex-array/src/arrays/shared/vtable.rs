// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

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
use crate::arrays::shared::SharedData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::scalar::Scalar;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

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

    fn array_hash<H: std::hash::Hasher>(array: &Array<Self>, state: &mut H, precision: Precision) {
        let current = array.current_array_ref();
        current.array_hash(state, precision);
        array.dtype.hash(state);
    }

    fn array_eq(array: &Array<Self>, other: &Array<Self>, precision: Precision) -> bool {
        let current = array.current_array_ref();
        let other_current = other.current_array_ref();
        current.array_eq(other_current, precision) && array.dtype == other.dtype
    }

    fn nbuffers(_array: &Array<Self>) -> usize {
        0
    }

    fn buffer(_array: &Array<Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("SharedArray has no buffers")
    }

    fn buffer_name(_array: &Array<Self>, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &Array<Self>) -> usize {
        1
    }

    fn child(array: &Array<Self>, idx: usize) -> ArrayRef {
        match idx {
            0 => array.current_array_ref().clone(),
            _ => vortex_panic!("SharedArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &Array<Self>, idx: usize) -> String {
        match idx {
            0 => "source".to_string(),
            _ => vortex_panic!("SharedArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &Array<Self>) -> VortexResult<Self::Metadata> {
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

    fn with_children(array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_error::vortex_ensure!(
            children.len() == 1,
            "SharedArray expects exactly 1 child, got {}",
            children.len()
        );
        let child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        array.set_source(child);
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        array
            .get_or_compute(|source| source.clone().execute::<Canonical>(ctx))
            .map(ExecutionResult::done)
    }
}
impl OperationsVTable<Shared> for Shared {
    fn scalar_at(
        array: &Array<Shared>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.current_array_ref().scalar_at(index)
    }
}

impl ValidityVTable<Shared> for Shared {
    fn validity(array: &Array<Shared>) -> VortexResult<Validity> {
        array.current_array_ref().validity()
    }
}
