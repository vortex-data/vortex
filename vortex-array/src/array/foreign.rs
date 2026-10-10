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
use vortex_session::registry::CachedId;

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
    /// The serialized encoding ID that no registered plugin could decode.
    encoding_id: ArrayId,
    metadata: Vec<u8>,
    buffers: Vec<BufferHandle>,
}

impl ForeignArrayData {
    pub fn new(encoding_id: ArrayId, metadata: Vec<u8>, buffers: Vec<BufferHandle>) -> Self {
        Self {
            encoding_id,
            metadata,
            buffers,
        }
    }

    /// Returns the encoding ID of the undecoded array this placeholder stands in for.
    pub fn encoding_id(&self) -> ArrayId {
        self.encoding_id
    }
}

impl Display for ForeignArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ForeignArrayData({}, {}B)",
            self.encoding_id,
            self.metadata.len()
        )
    }
}

impl ArrayHash for ForeignArrayData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: crate::EqMode) {
        self.encoding_id.as_ref().hash(state);
        self.metadata.hash(state);
        self.buffers.len().hash(state);
        for buffer in &self.buffers {
            buffer.array_hash(state, accuracy);
        }
    }
}

impl ArrayEq for ForeignArrayData {
    fn array_eq(&self, other: &Self, accuracy: crate::EqMode) -> bool {
        self.encoding_id == other.encoding_id
            && self.metadata == other.metadata
            && self.buffers.len() == other.buffers.len()
            && self
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(lhs, rhs)| lhs.array_eq(rhs, accuracy))
    }
}

/// Placeholder for an array whose encoding is not registered in the session.
///
/// Every foreign array reports the same [`ForeignArray`] ID, so an encoding ID always identifies
/// a single vtable type; the undecoded encoding ID is kept in [`ForeignArrayData`].
#[derive(Clone, Debug)]
pub struct ForeignArray;

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

    #[inline]
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.foreign");
        *ID
    }

    #[inline]
    fn static_id() -> Option<ArrayId> {
        Some(Self.id())
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            ForeignArrayData::new(array.encoding_id, array.metadata.clone(), buffers.to_vec()),
            array.slots().iter().cloned().collect(),
        ))
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone()))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        // Foreign arrays are built by `new_foreign_array` from the undecoded serialized ID; the
        // vtable is never registered as a plugin, so there is no ID to deserialize from here.
        vortex_bail!("Foreign arrays cannot be deserialized directly")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        format!("child[{idx}]")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        vortex_bail!(
            "Cannot execute unknown array encoding '{}'",
            array.data().encoding_id()
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
    Ok(Array::<ForeignArray>::try_from_parts(ArrayParts::new(
        ForeignArray,
        dtype,
        len,
        ForeignArrayData::new(id, metadata, buffers),
        children,
    ))?
    .into_array())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::ForeignArray;
    use super::new_foreign_array;
    use crate::VortexSessionExecute;
    use crate::array::VTable;
    use crate::arrays::Primitive;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn foreign_array_reports_its_own_id() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let foreign =
            new_foreign_array(Primitive.id(), dtype, 3, vec![], vec![], Default::default())?;

        assert_eq!(foreign.encoding_id(), ForeignArray.id());
        assert_eq!(
            foreign.as_::<ForeignArray>().data().encoding_id(),
            Primitive.id()
        );
        assert!(!foreign.is::<Primitive>());

        let mut ctx = crate::array_session().create_execution_ctx();
        let err = foreign
            .execute::<crate::Canonical>(&mut ctx)
            .expect_err("foreign arrays cannot execute");
        assert!(err.to_string().contains("vortex.primitive"), "{err}");
        Ok(())
    }
}
