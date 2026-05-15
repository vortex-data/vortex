// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::array::child_to_validity;
use crate::arrays::struct_::array::FIELDS_OFFSET;
use crate::arrays::struct_::array::VALIDITY_SLOT;
use crate::arrays::struct_::array::make_struct_slots;
use crate::arrays::struct_::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
mod kernel;
mod operations;
mod validity;

use vortex_session::registry::CachedId;

use crate::array::ArrayId;

/// A [`Struct`]-encoded Vortex array.
pub type StructArray = Array<Struct>;

impl VTable for Struct {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.struct");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let expected_slots = struct_dtype.nfields() + 1;
        if slots.len() != expected_slots {
            vortex_bail!(
                InvalidArgument: "StructArray has {} slots but expected {}",
                slots.len(),
                expected_slots
            );
        }

        let validity = child_to_validity(slots[VALIDITY_SLOT].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len()
            && validity_len != len
        {
            vortex_bail!(
                InvalidArgument: "StructArray validity length {} does not match outer length {}",
                validity_len,
                len
            );
        }

        let field_slots = &slots[FIELDS_OFFSET..];
        if field_slots.is_empty() {
            return Ok(());
        }

        for (idx, (slot, field_dtype)) in field_slots.iter().zip(struct_dtype.fields()).enumerate()
        {
            let field = slot
                .as_ref()
                .ok_or_else(|| vortex_error::vortex_err!("StructArray missing field slot {idx}"))?;
            if field.len() != len {
                vortex_bail!(
                    InvalidArgument: "StructArray field {idx} has length {} but expected {}",
                    field.len(),
                    len
                );
            }
            if field.dtype() != &field_dtype {
                vortex_bail!(
                    InvalidArgument: "StructArray field {idx} has dtype {} but expected {}",
                    field.dtype(),
                    field_dtype
                );
            }
        }

        Ok(())
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StructArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("StructArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if !metadata.is_empty() {
            vortex_bail!(
                "StructArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (Validity::from(*nullability), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            (Validity::Array(validity), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let field_children: Vec<_> = (0..struct_dtype.nfields())
            .map(|i| {
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                children.get(non_data_children + i, &child_dtype, len)
            })
            .try_collect()?;

        let slots = make_struct_slots(&field_children, &validity, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(slots))
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == VALIDITY_SLOT {
            "validity".to_string()
        } else {
            array.dtype().as_struct_fields().names()[idx - FIELDS_OFFSET].to_string()
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct Struct;
