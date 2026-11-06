// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayDeserialization;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::array::child_to_validity;
use crate::array::with_empty_buffers;
use crate::arrays::struct_::array::StructSlots;
use crate::arrays::struct_::array::struct_slots_with_capacity;
use crate::arrays::struct_::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::StructBuilder;
use crate::dtype::DType;
use crate::validity::Validity;
mod kernel;
mod operations;
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
                "StructArray has {} slots but expected {}",
                slots.len(),
                expected_slots
            );
        }

        let validity = child_to_validity(slots[StructSlots::VALIDITY].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len()
            && validity_len != len
        {
            vortex_bail!(
                "StructArray validity length {} does not match outer length {}",
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
                    "StructArray field {idx} has length {} but expected {}",
                    field.len(),
                    len
                );
            }
            if field.dtype() != &field_dtype {
                vortex_bail!(
                    "StructArray field {idx} has dtype {} but expected {}",
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

impl ArrayPlugin for Struct {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;

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

        let mut slots = struct_slots_with_capacity(&validity, len, struct_dtype.nfields());
        for i in 0..struct_dtype.nfields() {
            let child_dtype = struct_dtype
                .field_by_index(i)
                .vortex_expect("no out of bounds");
            slots.push(Some(children.get(
                non_data_children + i,
                &child_dtype,
                len,
            )?));
        }

        Ok(Array::<Self>::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            EmptyArrayData,
            slots,
        ))?
        .into_array())
    }
}

#[derive(Clone, Debug)]
pub struct Struct;
