// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionStep;
use vortex::array::Precision;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::stats::StatsSetRef;
use vortex::array::vtable;
use vortex::array::vtable::ArrayId;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityVTableFromChild;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::session::VortexSession;

use crate::encodings::norm::array::NormVectorArray;

mod operations;
mod validity;

vtable!(NormVector);

#[derive(Debug)]
pub struct NormVector;

impl VTable for NormVector {
    type Array = NormVectorArray;
    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(_array: &NormVectorArray) -> ArrayId {
        ArrayId::new_ref("vortex.tensor.norm_vector")
    }

    fn len(array: &NormVectorArray) -> usize {
        array.vector_array().len()
    }

    fn dtype(array: &NormVectorArray) -> &DType {
        array.vector_array().dtype()
    }

    fn stats(array: &NormVectorArray) -> StatsSetRef<'_> {
        array.vector_array().statistics()
    }

    fn array_hash<H: Hasher>(array: &NormVectorArray, state: &mut H, precision: Precision) {
        todo!()
    }

    fn array_eq(array: &NormVectorArray, other: &NormVectorArray, precision: Precision) -> bool {
        todo!()
    }

    fn nbuffers(array: &NormVectorArray) -> usize {
        todo!()
    }

    fn buffer(array: &NormVectorArray, idx: usize) -> BufferHandle {
        todo!()
    }

    fn buffer_name(array: &NormVectorArray, idx: usize) -> Option<String> {
        todo!()
    }

    fn nchildren(array: &NormVectorArray) -> usize {
        todo!()
    }

    fn child(array: &NormVectorArray, idx: usize) -> ArrayRef {
        todo!()
    }

    fn child_name(array: &NormVectorArray, idx: usize) -> String {
        todo!()
    }

    fn metadata(array: &NormVectorArray) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        todo!()
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<NormVectorArray> {
        todo!()
    }

    fn with_children(array: &mut NormVectorArray, children: Vec<ArrayRef>) -> VortexResult<()> {
        todo!()
    }

    fn execute(array: &NormVectorArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        todo!()
    }
}
