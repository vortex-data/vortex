// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Narrow;
use super::NarrowArraySlotsExt;
use super::NarrowSlots;
use super::compute::PARENT_RULES;
use super::validate_dtypes;
use crate::Array;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayView;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::VTable;
use crate::array::with_empty_buffers;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::vtable::OperationsVTable;
use crate::vtable::ValidityChild;
use crate::vtable::ValidityVTableFromChild;

impl VTable for Narrow {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.narrow");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(slots.len() == 1, "NarrowArray requires one child");
        let values = slots[0]
            .as_ref()
            .ok_or_else(|| vortex_err!("NarrowArray requires a values child"))?;
        validate_dtypes(values.dtype(), dtype)?;
        vortex_ensure!(values.len() == len, "NarrowArray child length must match");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("NarrowArray buffer index {idx} out of bounds")
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![PType::try_from(array.values().dtype())? as u8]))
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
        vortex_ensure!(buffers.is_empty(), "NarrowArray expects no buffers");
        vortex_ensure!(children.len() == 1, "NarrowArray requires one child");
        let [storage_ptype] = metadata else {
            return Err(vortex_err!(
                "NarrowArray expects one storage-type metadata byte"
            ));
        };
        let storage_ptype = PType::try_from(i32::from(*storage_ptype))
            .map_err(|err| vortex_err!("Invalid NarrowArray storage type: {err}"))?;
        let storage_dtype = DType::Primitive(storage_ptype, dtype.nullability());
        validate_dtypes(&storage_dtype, dtype)?;
        let values = children.get(0, &storage_dtype, len)?;
        Ok(ArrayParts::new(Self, dtype.clone(), len, EmptyArrayData)
            .with_slots(NarrowSlots { values }.into_slots()))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        NarrowSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Yield the widening cast so the child's own cast kernel can fuse decoding and widening.
        Ok(ExecutionResult::done(
            array.values().cast(array.dtype().clone())?,
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<Narrow> for Narrow {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array
            .values()
            .execute_scalar(index, ctx)?
            .cast(array.dtype())
    }
}

impl ValidityChild<Narrow> for Narrow {
    fn validity_child(array: ArrayView<'_, Self>) -> ArrayRef {
        array.values().clone()
    }
}
