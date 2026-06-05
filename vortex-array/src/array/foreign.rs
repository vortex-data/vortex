// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

#[derive(Clone, Debug)]
pub struct ForeignArrayData {
    metadata: Vec<u8>,
    buffers: Vec<BufferHandle>,
}

impl ForeignArrayData {
    pub fn new(metadata: Vec<u8>, buffers: Vec<BufferHandle>) -> Self {
        Self { metadata, buffers }
    }
}

impl Display for ForeignArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ForeignArrayData({}B)", self.metadata.len())
    }
}

impl ArrayHash for ForeignArrayData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: crate::EqMode) {
        self.metadata.hash(state);
        self.buffers.len().hash(state);
        for buffer in &self.buffers {
            buffer.array_hash(state, accuracy);
        }
    }
}

impl ArrayEq for ForeignArrayData {
    fn array_eq(&self, other: &Self, accuracy: crate::EqMode) -> bool {
        self.metadata == other.metadata
            && self.buffers.len() == other.buffers.len()
            && self
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(lhs, rhs)| lhs.array_eq(rhs, accuracy))
    }
}

#[derive(Clone, Debug)]
pub struct ForeignArray {
    id: ArrayId,
}

impl ForeignArray {
    pub fn new(id: ArrayId) -> Self {
        Self { id }
    }
}

pub struct ForeignValidityVTable;

impl ValidityVTable<ForeignArray> for ForeignValidityVTable {
    fn validity(array: ArrayView<'_, ForeignArray>) -> VortexResult<Validity> {
        Ok(Validity::from(array.dtype().nullability()))
    }
}

impl VTable for ForeignArray {
    type TypedArrayData = ForeignArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ForeignValidityVTable;

    fn id(&self) -> ArrayId {
        self.id
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        _dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        Ok(())
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.buffers.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        array.buffers[idx].clone()
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(format!("buffer[{idx}]"))
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let child_arrays = (0..children.len())
            .map(|idx| children.get(idx, dtype, len).map(Some))
            .collect::<VortexResult<ArraySlots>>()?;

        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ForeignArrayData::new(metadata.to_vec(), buffers.to_vec()),
        )
        .with_slots(child_arrays))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        format!("child[{idx}]")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        vortex_bail!(
            "Cannot execute unknown array encoding '{}'",
            array.encoding_id()
        )
    }
}

pub fn new_foreign_array(
    id: ArrayId,
    dtype: DType,
    len: usize,
    metadata: Vec<u8>,
    buffers: Vec<BufferHandle>,
    children: ArraySlots,
) -> VortexResult<ArrayRef> {
    Ok(Array::<ForeignArray>::try_from_parts(
        ArrayParts::new(
            ForeignArray::new(id),
            dtype,
            len,
            ForeignArrayData::new(metadata, buffers),
        )
        .with_slots(children),
    )?
    .into_array())
}
