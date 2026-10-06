// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validate native wide integer storage and expose its canonical byte representation.
//!
//! The buffer width must match the integer dtype. Canonical execution shares the same buffer as
//! a fixed-size list of bytes; scalar access preserves signed numeric values.

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::WideIntegerArray;
use super::WideIntegerArrayExt;
use super::WideIntegerData;
use super::WideIntegerEncoding;
use super::WideIntegerSlots;
use crate::Array;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::PType;
use crate::dtype::integer::integer_dtype;
use crate::dtype::integer::signed_integer_type;
use crate::integer::scalar_from_integer;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::OperationsVTable;
use crate::vtable::ValidityVTable;

impl VTable for WideIntegerEncoding {
    type TypedArrayData = WideIntegerData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.integer.buffer");
        *ID
    }

    fn validate(
        &self,
        data: &WideIntegerData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(data.values_type, DecimalType::I128 | DecimalType::I256),
            "Expected wide integer storage"
        );
        let alignment =
            match_each_decimal_value_type!(data.values_type, |T| { Alignment::of::<T>() });
        vortex_ensure!(
            data.values.is_aligned_to(alignment),
            "Integer storage is not aligned to {alignment:?}"
        );
        vortex_ensure_eq!(slots.len(), 1);
        let expected_dtype = integer_dtype(data.values_type, dtype.nullability());
        vortex_ensure_eq!(dtype, &expected_dtype);
        vortex_ensure!(
            data.values
                .len()
                .is_multiple_of(data.values_type.byte_width()),
            "Integer buffer must contain whole values"
        );
        vortex_ensure_eq!(data.values.len() / data.values_type.byte_width(), len);
        if let Some(validity) = &slots[0] {
            vortex_ensure!(
                dtype.is_nullable(),
                "Expected nullable dtype with a validity child, got {dtype}"
            );
            vortex_ensure_eq!(validity.dtype(), &Validity::DTYPE);
            vortex_ensure_eq!(validity.len(), len);
        }

        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        fixed_width::buffer("WideInteger", array.buffer_handle(), idx)
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        fixed_width::buffer_name(idx)
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let rebuilt = WideIntegerArray::try_new_handle(
            fixed_width::single_buffer(buffers)?,
            array.values_type(),
            array.integer_validity(),
        )?;
        Ok(ArrayParts::new(
            Self,
            rebuilt.dtype().clone(),
            rebuilt.len(),
            rebuilt.data().clone(),
            rebuilt.slots().iter().cloned().collect(),
        ))
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
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(metadata.len(), 0);
        let values_type = signed_integer_type(dtype)
            .ok_or_else(|| vortex_err!("Expected a wide integer dtype, got {dtype}"))?;
        let array = WideIntegerArray::try_new_handle(
            fixed_width::single_buffer(buffers)?,
            values_type,
            fixed_width::deserialize_validity(dtype.nullability(), len, children)?,
        )?;
        vortex_ensure_eq!(array.len(), len);
        Ok(ArrayParts::new(
            Self,
            array.dtype().clone(),
            array.len(),
            array.data().clone(),
            array.slots().iter().cloned().collect(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        super::compute::RULES.evaluate(array, parent, child_idx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        WideIntegerSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let bytes = PrimitiveArray::from_buffer_handle(
            array.buffer_handle().clone(),
            PType::U8,
            Validity::NonNullable,
        );
        let storage = FixedSizeListArray::try_new(
            bytes.into_array(),
            u32::try_from(array.values_type().byte_width())
                .vortex_expect("DecimalType is at most 32 bytes"),
            array.integer_validity(),
            array.len(),
        )?;
        Ok(ExecutionResult::done(ExtensionArray::new(
            array.dtype().as_extension().clone(),
            storage.into_array(),
        )))
    }
}

impl OperationsVTable<WideIntegerEncoding> for WideIntegerEncoding {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = match_each_decimal_value_type!(array.values_type(), |T| {
            let values = Buffer::<T>::from_byte_buffer(array.buffer_handle().as_host().clone());
            DecimalValue::from(values[index])
        });
        scalar_from_integer(value, array.dtype())
    }
}

impl ValidityVTable<WideIntegerEncoding> for WideIntegerEncoding {
    fn validity(array: ArrayView<'_, WideIntegerEncoding>) -> VortexResult<Validity> {
        Ok(array.integer_validity())
    }
}
