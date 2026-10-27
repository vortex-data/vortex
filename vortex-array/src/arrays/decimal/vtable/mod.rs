// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use prost::Message;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::decimal::DecimalData;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::serde::ArrayChildren;
mod kernel;
mod operations;
mod validity;

use std::hash::Hash;

use vortex_session::registry::CachedId;

use crate::EqMode;
use crate::array::ArrayId;
use crate::arrays::decimal::array::DecimalSlots;
use crate::arrays::decimal::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
/// A [`Decimal`]-encoded Vortex array.
///
/// The physical storage type is no wider than [`DecimalType::smallest_decimal_value_type`]
/// for the precision; smaller storage is allowed when the values fit. See [`DecimalData`]
/// for the full storage contract.
pub type DecimalArray = Array<Decimal>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

// The type of the values can be determined by looking at the type info...right?
#[derive(prost::Message)]
pub struct DecimalMetadata {
    #[prost(int32, tag = "1")]
    pub(super) values_type: i32,
}

impl DecimalMetadata {
    fn get_values_type(&self) -> VortexResult<DecimalType> {
        DecimalType::try_from(self.values_type).map_err(|_| {
            vortex_err!(
                "DecimalMetadata values_type must be a known DecimalType, got {}",
                self.values_type
            )
        })
    }
}

impl ArrayHash for DecimalData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.values.array_hash(state, accuracy);
        std::mem::discriminant(&self.values_type).hash(state);
    }
}

impl ArrayEq for DecimalData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.values.array_eq(&other.values, accuracy) && self.values_type == other.values_type
    }
}

impl VTable for Decimal {
    type TypedArrayData = DecimalData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        fixed_width::buffer("DecimalArray", &array.values, idx)
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        fixed_width::buffer_name(idx)
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        data.values = fixed_width::single_buffer(buffers)?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
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
        let DType::Decimal(decimal_dtype, nullability) = dtype else {
            vortex_bail!("Expected decimal dtype, got {dtype:?}");
        };
        vortex_ensure_eq!(
            data.decimal_dtype,
            *decimal_dtype,
            "DecimalArray data dtype does not match outer dtype",
        );
        vortex_ensure!(
            data.values_type <= DecimalType::smallest_decimal_value_type(decimal_dtype),
            "DecimalArray storage exceeds the type required by its precision",
        );
        vortex_ensure_eq!(
            data.len(),
            len,
            "DecimalArray length does not match outer length",
        );
        let validity =
            crate::array::child_to_validity(slots[DecimalSlots::VALIDITY].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(
                validity_len,
                len,
                "DecimalArray validity len does not match outer length",
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
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = DecimalMetadata::decode(metadata)?;
        let values_type = metadata.get_values_type()?;
        let values = fixed_width::single_buffer(buffers)?;

        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;

        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Expected Decimal dtype, got {:?}", dtype)
        };

        let slots = DecimalData::make_slots(&validity, len);
        let data = match_each_decimal_value_type!(values_type, |D| {
            // Check and reinterpret-cast the buffer
            vortex_ensure!(
                values.is_aligned_to(Alignment::of::<D>()),
                "DecimalArray buffer not aligned for values type {:?}",
                D::DECIMAL_TYPE
            );
            // Older writers allowed storage wider than the precision requires.
            let (values, values_type) =
                DecimalData::narrow_to_precision(values, values_type, *decimal_dtype)?;
            DecimalData::try_new_handle(values, values_type, *decimal_dtype)
        })?;
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))
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
            vortex_bail!("append_to_builder for Decimal requires a DecimalBuilder");
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

#[derive(Clone, Debug)]
pub struct Decimal;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::DecimalMetadata;
    use crate::ArrayContext;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array::VTable;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_array_serde() {
        let array = DecimalArray::new(
            buffer![100i64, 200i64, 300i64, 400i64, 500i64],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();

        let array_ctx = ArrayContext::empty();
        let out = array
            .into_array()
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())
            .unwrap();
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(&dtype, 5, &ReadContext::new(array_ctx.to_ids()), &SESSION)
            .unwrap();
        assert!(decoded.is::<Decimal>());
        assert_arrays_eq!(
            decoded,
            DecimalArray::from_iter([100i128, 200, 300, 400, 500], DecimalDType::new(10, 2)),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn deserialize_legacy_oversized_storage() {
        let array = DecimalArray::new(
            buffer![100i64, -200, 300],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let array_ctx = ArrayContext::empty();
        let buffers = array
            .into_array()
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())
            .unwrap();
        let mut bytes = ByteBufferMut::empty();
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        // Older writers allowed i64 storage with precision 3. Decode the same wire storage
        // with that logical dtype and ensure it is normalized before reaching kernels.
        let dtype = DecimalDType::new(3, 2);
        let parts = SerializedArray::try_from(bytes.freeze()).unwrap();
        let decoded = parts
            .decode(
                &DType::Decimal(dtype, Nullability::NonNullable),
                3,
                &ReadContext::new(array_ctx.to_ids()),
                &SESSION,
            )
            .unwrap();
        assert_eq!(decoded.as_::<Decimal>().values_type(), DecimalType::I16);
        assert_arrays_eq!(
            decoded,
            DecimalArray::new(buffer![100i16, -200, 300], dtype, Validity::NonNullable),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_nullable_decimal_serde_roundtrip() {
        let mut ctx = SESSION.create_execution_ctx();
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
            .serialize(&array_ctx, &SESSION, &SerializeOptions::default())
            .unwrap();
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &SESSION)
            .unwrap();

        assert_arrays_eq!(decoded, array, &mut ctx);
    }

    #[rstest]
    #[case::first_unused(6)]
    #[case::negative(-1)]
    #[case::max(i32::MAX)]
    fn test_deserialize_rejects_unknown_values_type(#[case] raw: i32) {
        let metadata = DecimalMetadata { values_type: raw }.encode_to_vec();
        let buffers = [BufferHandle::new_host(ByteBuffer::zeroed(4))];
        let children: Vec<ArrayRef> = Vec::new();

        let result = Decimal.deserialize(
            &DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable),
            4,
            &metadata,
            &buffers,
            &children,
            &SESSION,
        );

        let expected =
            format!("DecimalMetadata values_type must be a known DecimalType, got {raw}");
        assert!(result.is_err_and(|err| err.to_string().contains(&expected)));
    }

    #[rstest]
    #[case::i8(
        DecimalArray::from_iter([1i8, -2, 3], DecimalDType::new(2, 0)),
        DecimalType::I8
    )]
    #[case::i16(
        DecimalArray::from_iter([1i16, -2, 3], DecimalDType::new(4, 0)),
        DecimalType::I16
    )]
    #[case::i32(
        DecimalArray::from_iter([1i32, -2, 3], DecimalDType::new(9, 2)),
        DecimalType::I32
    )]
    #[case::i64(
        DecimalArray::from_iter([1i64, -2, 3], DecimalDType::new(18, 2)),
        DecimalType::I64
    )]
    #[case::i128(
        DecimalArray::from_iter([1i128, -2, 3], DecimalDType::new(38, 2)),
        DecimalType::I128
    )]
    #[case::i256(
        DecimalArray::from_iter(
            [i256::from_i128(1), i256::from_i128(-2), i256::from_i128(3)],
            DecimalDType::new(76, 2),
        ),
        DecimalType::I256
    )]
    fn test_serde_each_values_type(
        #[case] array: DecimalArray,
        #[case] values_type: DecimalType,
    ) -> VortexResult<()> {
        let dtype = array.dtype().clone();
        let len = array.len();

        let array_ctx = ArrayContext::empty();
        let out = array.clone().into_array().serialize(
            &array_ctx,
            &SESSION,
            &SerializeOptions::default(),
        )?;
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &SESSION)?;

        assert_eq!(decoded.as_::<Decimal>().values_type(), values_type);
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());

        Ok(())
    }
}
