// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::Precision;
use vortex::array::RawMetadata;
use vortex::array::SerializeMetadata;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::stats::StatsSetRef;
use vortex::array::validity::Validity;
use vortex::array::vtable;
use vortex::array::vtable::Array;
use vortex::array::vtable::ArrayId;
use vortex::array::vtable::OperationsVTable;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityVTable;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

use crate::arrays::py::PythonArray;

vtable!(Python, PythonVTable);

/// Wrapper struct encapsulating a Python encoding.
#[derive(Debug, Clone)]
pub struct PythonVTable {
    pub id: ArrayId,
}

impl VTable for PythonVTable {
    type Array = PythonArray;

    type Metadata = RawMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(array: &Self::Array) -> &Self {
        &array.vtable
    }

    fn id(&self) -> ArrayId {
        self.id.clone()
    }

    fn len(array: &PythonArray) -> usize {
        array.len
    }

    fn dtype(array: &PythonArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PythonArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &PythonArray, state: &mut H, _precision: Precision) {
        Arc::as_ptr(&array.object).hash(state);
        array.vtable.id.hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
    }

    fn array_eq(array: &PythonArray, other: &PythonArray, _precision: Precision) -> bool {
        Arc::ptr_eq(&array.object, &other.object)
            && array.vtable.id == other.vtable.id // TODO(ngates): in the future this check is already done
            && array.len == other.len
            && array.dtype == other.dtype
    }

    fn nbuffers(_array: &PythonArray) -> usize {
        0
    }

    fn buffer(_array: &PythonArray, idx: usize) -> BufferHandle {
        vortex_panic!("PythonArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &PythonArray, _idx: usize) -> Option<String> {
        None
    }

    fn metadata(array: &PythonArray) -> VortexResult<Self::Metadata> {
        Python::attach(|py| {
            let obj = array.object.bind(py);
            if !obj
                .hasattr(intern!(py, "metadata"))
                .map_err(|e| vortex_err!("{}", e))?
            {
                // The class does not have a metadata attribute so does not support serialization.
                return Ok(RawMetadata(vec![]));
            }

            let bytes = obj
                .call_method(intern!(py, "__vx_metadata__"), (), None)
                .map_err(|e| vortex_err!("{}", e))?
                .cast::<PyBytes>()
                .map_err(|_| vortex_err!("Expected array metadata to be Python bytes"))?
                .as_bytes()
                .to_vec();

            Ok(RawMetadata(bytes))
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(RawMetadata(bytes.to_vec()))
    }

    fn build(
        _dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<PythonArray> {
        todo!()
    }

    fn slots(_array: &PythonArray) -> &[Option<ArrayRef>] {
        &[]
    }

    fn slot_name(_array: &PythonArray, idx: usize) -> String {
        vortex_panic!("PythonArray has no slots, requested index {idx}")
    }

    fn with_slots(_array: &mut PythonArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.is_empty(),
            "PythonArray has no slots, got {}",
            slots.len()
        );
        Ok(())
    }

    fn execute(_array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        todo!()
    }
}

impl OperationsVTable<PythonVTable> for PythonVTable {
    fn scalar_at(
        _array: &PythonArray,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        todo!()
    }
}

impl ValidityVTable<PythonVTable> for PythonVTable {
    fn validity(_array: &PythonArray) -> VortexResult<Validity> {
        todo!()
    }
}
