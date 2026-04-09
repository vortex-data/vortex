// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use kernel::PARENT_KERNELS;
use prost::Message;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::decimal::DecimalData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
mod kernel;
mod operations;
mod validity;

use std::hash::Hash;

use crate::Precision;
use crate::array::ArrayId;
use crate::arrays::decimal::array::SLOT_NAMES;
use crate::arrays::decimal::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
/// A [`Decimal`]-encoded Vortex array.
pub type DecimalArray = Array<Decimal>;

// The type of the values can be determined by looking at the type info...right?
#[derive(prost::Message)]
pub struct DecimalMetadata {
    #[prost(enumeration = "DecimalType", tag = "1")]
    pub(super) values_type: i32,
}

impl ArrayHash for DecimalData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.values.array_hash(state, precision);
        std::mem::discriminant(&self.values_type).hash(state);
    }
}

impl ArrayEq for DecimalData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.values.array_eq(&other.values, precision) && self.values_type == other.values_type
    }
}

impl VTable for Decimal {
    type ArrayData = DecimalData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.values.clone(),
            _ => vortex_panic!("DecimalArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("values".to_string()),
            _ => None,
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DecimalMetadata {
                values_type: array.values_type() as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn validate(
        &self,
        data: &DecimalData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Decimal(_, nullability) = dtype else {
            vortex_bail!("Expected decimal dtype, got {dtype:?}");
        };
        vortex_ensure!(
            data.len() == len,
            InvalidArgument:
            "DecimalArray length {} does not match outer length {}",
            data.len(),
            len
        );
        let validity = crate::array::child_to_validity(&slots[0], *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                InvalidArgument:
                "DecimalArray validity len {} does not match outer length {}",
                validity_len,
                len
            );
        }

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
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        let metadata = DecimalMetadata::decode(metadata)?;
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let values = buffers[0].clone();

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Expected Decimal dtype, got {:?}", dtype)
        };

        let slots = DecimalData::make_slots(&validity, len);
        let data = match_each_decimal_value_type!(metadata.values_type(), |D| {
            // Check and reinterpret-cast the buffer
            vortex_ensure!(
                values.is_aligned_to(Alignment::of::<D>()),
                "DecimalArray buffer not aligned for values type {:?}",
                D::DECIMAL_TYPE
            );
            DecimalData::try_new_handle(values, metadata.values_type(), *decimal_dtype)
        })?;
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
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
pub struct Decimal;

impl Decimal {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.decimal");
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::validity::Validity;

    #[test]
    fn test_array_serde() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 400i128, 500i128],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();

        let ctx = ArrayContext::empty();
        let out = array
            .into_array()
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
            .unwrap();
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(&dtype, 5, &ReadContext::new(ctx.to_ids()), &LEGACY_SESSION)
            .unwrap();
        assert!(decoded.is::<Decimal>());
    }

    #[test]
    fn test_nullable_decimal_serde_roundtrip() {
        let array = DecimalArray::new(
            buffer![1234567i32, 0i32, -9999999i32],
            DecimalDType::new(7, 3),
            Validity::from_iter([true, false, true]),
        );
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let out = array
            .clone()
            .into_array()
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
            .unwrap();
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert_arrays_eq!(decoded, array);
    }
}
