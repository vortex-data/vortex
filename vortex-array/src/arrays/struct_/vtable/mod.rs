// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::array::with_empty_buffers;
use crate::arrays::struct_::array::StructSlots;
use crate::arrays::struct_::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::StructBuilder;
use crate::dtype::DType;
mod kernel;
mod operations;
mod plugin;
mod validity;

use vortex_session::registry::CachedId;

use crate::array::ArrayId;

/// A [`Struct`]-encoded Vortex array.
pub type StructArray = Array<Struct>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

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

        let validity = child_to_validity(slots[StructSlots::VALIDITY].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len()
            && validity_len != len
        {
            vortex_bail!(
                InvalidArgument: "StructArray validity length {} does not match outer length {}",
                validity_len,
                len
            );
        }

        let field_slots = &slots[StructSlots::FIELDS_OFFSET..];
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == StructSlots::VALIDITY {
            "validity".to_string()
        } else {
            array.dtype().as_struct_fields().names()[idx - StructSlots::FIELDS_OFFSET].to_string()
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<StructBuilder>() else {
            vortex_bail!("append_to_builder for Struct requires a StructBuilder");
        };
        builder.append_struct_array(&array.into_owned(), ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct Struct;
