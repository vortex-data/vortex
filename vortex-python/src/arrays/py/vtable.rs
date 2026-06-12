// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex::array::Array;
use vortex::array::ArrayEq;
use vortex::array::ArrayHash;
use vortex::array::ArrayId;
use vortex::array::ArrayParts;
use vortex::array::ArrayRef;
use vortex::array::ArrayView;
use vortex::array::EqMode;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::OperationsVTable;
use vortex::array::VTable;
use vortex::array::ValidityVTable;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_panic;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

use crate::arrays::py::PythonArray;

/// Wrapper struct encapsulating a Python encoding.
#[derive(Debug, Clone)]
pub struct PythonVTable {
    pub id: ArrayId,
}

impl ArrayHash for PythonArray {
    fn array_hash<H: std::hash::Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        Arc::as_ptr(&self.object).hash(state);
    }
}

impl ArrayEq for PythonArray {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        Arc::ptr_eq(&self.object, &other.object)
    }
}

impl VTable for PythonVTable {
    type TypedArrayData = PythonArray;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        self.id
    }

    fn validate(
        &self,
        data: &PythonArray,
        dtype: &DType,
        len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(data.vtable.id == self.id, "PythonArray vtable id mismatch");
        vortex_ensure!(&data.dtype == dtype, "PythonArray dtype mismatch");
        vortex_ensure!(data.len == len, "PythonArray len mismatch");
        Ok(())
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
        bytes: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        _ = bytes;
        vortex_bail!("PythonArray deserialization is not supported");
    }

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        vortex_panic!("PythonArray has no slots")
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
