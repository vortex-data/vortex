// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validate and execute the canonical decimal child representation.
//!
//! Legacy buffer metadata is decoded here for [`DecimalPlugin`](super::DecimalPlugin). The
//! canonical array has no buffers, and its integer child owns both values and validity.

mod kernel;
mod operations;
mod validity;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Decimal;
use super::DecimalArray;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::arrays::decimal::array::DecimalSlots;
use crate::arrays::decimal::compute::rules::RULES;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::integer::integer_dtype;
use crate::serde::ArrayChildren;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

/// Metadata used by the historical buffer-backed decimal wire representation.
#[derive(prost::Message)]
pub struct DecimalMetadata {
    /// Native signed width used by the serialized values buffer.
    #[prost(enumeration = "DecimalType", tag = "1")]
    pub(super) values_type: i32,
}

impl VTable for Decimal {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Decimal has no buffers, requested {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.is_empty(), "Decimal has no buffers");
        Ok(
            ArrayParts::new(Self, array.dtype().clone(), array.len(), EmptyArrayData)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // The plugin writes the historical buffer representation instead of this child tree.
        Ok(None)
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Decimal(decimal_dtype, nullability) = dtype else {
            vortex_bail!("Expected a decimal dtype, got {dtype}");
        };
        vortex_ensure!(slots.len() == 1, "Decimal requires one integer child");
        let Some(values) = &slots[DecimalSlots::VALUES] else {
            vortex_bail!("Decimal requires an integer child");
        };
        let values_dtype = integer_dtype(
            DecimalType::smallest_decimal_value_type(decimal_dtype),
            *nullability,
        );
        vortex_ensure!(
            values.dtype() == &values_dtype,
            "Decimal {dtype} requires integer child {values_dtype}, got {}",
            values.dtype()
        );
        vortex_ensure!(
            values.len() == len,
            "Decimal length {len} does not match child length {}",
            values.len()
        );
        Ok(())
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
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Expected a decimal dtype, got {dtype}");
        };
        let metadata = DecimalMetadata::decode(metadata)?;
        let values_type = DecimalType::try_from(metadata.values_type)
            .map_err(|err| vortex_err!("Invalid decimal storage type: {err}"))?;
        let values = fixed_width::single_buffer(buffers)?;
        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;
        let array = DecimalArray::try_new_handle(values, values_type, *decimal_dtype, validity)?;
        vortex_ensure!(
            array.len() == len,
            "Decimal buffer length {} does not match declared length {len}",
            array.len()
        );
        Ok(ArrayParts::new(Self, dtype.clone(), len, EmptyArrayData)
            .with_slots(array.slots().iter().cloned().collect()))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DecimalSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<DecimalBuilder>() else {
            vortex_bail!("Decimal requires a DecimalBuilder");
        };
        builder.append_decimal_array(&array.into_owned(), ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::validity::Validity;

    #[test]
    fn test_array_serde() {
        let session = array_session();
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 400i128, 500i128],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();

        let array_ctx = ArrayContext::empty();
        let out = array
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())
            .unwrap();
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(&dtype, 5, &ReadContext::new(array_ctx.to_ids()), &session)
            .unwrap();
        assert!(decoded.is::<Decimal>());
    }

    #[test]
    fn test_nullable_decimal_serde_roundtrip() {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let array = DecimalArray::new(
            buffer![1234567i32, 0i32, -9999999i32],
            DecimalDType::new(7, 3),
            Validity::from_iter([true, false, true]),
        );
        let dtype = array.dtype().clone();
        let len = array.len();

        let array_ctx = ArrayContext::empty();
        let out = array
            .clone()
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())
            .unwrap();
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)
            .unwrap();

        assert_arrays_eq!(decoded, array, &mut ctx);
    }
}
