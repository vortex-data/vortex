// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use vortex::array::Array;
use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayView;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::OperationsVTable;
use vortex::array::Precision;
use vortex::array::RawMetadata;
use vortex::array::SerializeMetadata;
use vortex::array::VTable;
use vortex::array::ValidityVTable;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::stats::ArrayStats;
use vortex::array::validity::Validity;
use vortex::array::vtable;
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
    type ArrayData = PythonArray;

    type Metadata = RawMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(array: &Self::ArrayData) -> &Self {
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

    fn stats(array: &PythonArray) -> &ArrayStats {
        &array.stats
    }

    fn array_hash<H: std::hash::Hasher>(array: &PythonArray, state: &mut H, _precision: Precision) {
        Arc::as_ptr(&array.object).hash(state);
    }

    fn array_eq(array: &PythonArray, other: &PythonArray, _precision: Precision) -> bool {
        Arc::ptr_eq(&array.object, &other.object)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("PythonArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn child(_array: ArrayView<'_, Self>, idx: usize) -> ArrayRef {
        vortex_panic!("PythonArray child index {idx} out of bounds")
    }

    fn child_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("PythonArray child_name index {idx} out of bounds")
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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

    fn slots(_array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &[]
    }

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        vortex_panic!("PythonArray has no slots")
    }

    fn with_slots(_array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.is_empty(),
            "PythonArray has no slots, got {}",
            slots.len()
        );
        Ok(())
    }

    fn execute(_array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        todo!()
    }
}

impl OperationsVTable<PythonVTable> for PythonVTable {
    fn scalar_at(
        _array: ArrayView<'_, PythonVTable>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        todo!()
    }
}

impl ValidityVTable<PythonVTable> for PythonVTable {
    fn validity(_array: ArrayView<'_, PythonVTable>) -> VortexResult<Validity> {
        todo!()
    }
}
