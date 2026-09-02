// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::ops::Range;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BlockResidual;
use crate::BlockResidualCodec;
use crate::BlockResidualEstimate;
use crate::block_residual_array::decompress_ordered_f16;
use crate::block_residual_array::decompress_ordered_f32;
use crate::block_residual_array::decompress_ordered_f64;
use crate::codec::BlockResidualCodecEstimate;

/// IEEE floats mapped to unsigned integers that preserve numeric order.
pub type OrderedFloatArray = Array<OrderedFloat>;

#[array_slots(OrderedFloat)]
pub struct OrderedFloatSlots {
    /// Ordered unsigned float bits.
    #[slot(0)]
    pub encoded: ArrayRef,
}

#[derive(Clone, Debug, Default)]
pub struct OrderedFloatData;

impl Display for OrderedFloatData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArrayHash for OrderedFloatData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for OrderedFloatData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct OrderedFloat;

impl VTable for OrderedFloat {
    type TypedArrayData = OrderedFloatData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.ordered_float");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let ptype = PType::try_from(dtype)?;
        vortex_ensure!(
            matches!(ptype, PType::F16 | PType::F32 | PType::F64),
            "OrderedFloatArray requires f16, f32, or f64"
        );
        let encoded = OrderedFloatSlotsView::from_slots(slots).encoded;
        let expected = DType::Primitive(ordered_ptype(ptype)?, dtype.nullability());
        vortex_ensure!(
            encoded.dtype() == &expected,
            "OrderedFloatArray expected child dtype {expected}, got {}",
            encoded.dtype()
        );
        vortex_ensure!(
            encoded.len() == len,
            "OrderedFloatArray child length differs"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("OrderedFloatArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("OrderedFloatArray buffer_name {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
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
        vortex_ensure!(buffers.is_empty(), "OrderedFloatArray expects no buffers");
        vortex_ensure!(
            metadata.is_empty(),
            "OrderedFloatArray metadata must be empty"
        );
        vortex_ensure!(children.len() == 1, "OrderedFloatArray requires one child");
        let ptype = PType::try_from(dtype)?;
        let child_dtype = DType::Primitive(ordered_ptype(ptype)?, dtype.nullability());
        let encoded = children.get(0, &child_dtype, len)?;
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, OrderedFloatData)
                .with_slots(OrderedFloatSlots { encoded }.into_slots()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        OrderedFloatSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let decoded = if let Some(block_residual) = array.encoded().as_typed::<BlockResidual>() {
            match array.dtype().as_ptype() {
                PType::F16 => decompress_ordered_f16(block_residual, ctx)?,
                PType::F32 => decompress_ordered_f32(block_residual, ctx)?,
                PType::F64 => decompress_ordered_f64(block_residual, ctx)?,
                ptype => vortex_bail!("unsupported OrderedFloat ptype {ptype}"),
            }
        } else {
            decode_primitive(array.as_view(), ctx)?
        };
        Ok(ExecutionResult::done(decoded.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<OrderedFloat> for OrderedFloat {
    fn scalar_at(
        array: ArrayView<'_, OrderedFloat>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let scalar = array.encoded().execute_scalar(index, ctx)?;
        if scalar.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        Ok(match array.dtype().as_ptype() {
            PType::F16 => Scalar::primitive(
                f16::from_bits(unordered_u16(
                    scalar
                        .as_primitive()
                        .typed_value::<u16>()
                        .vortex_expect("validated ordered float scalar"),
                )),
                array.dtype().nullability(),
            ),
            PType::F32 => Scalar::primitive(
                f32::from_bits(unordered_u32(
                    scalar
                        .as_primitive()
                        .typed_value::<u32>()
                        .vortex_expect("validated ordered float scalar"),
                )),
                array.dtype().nullability(),
            ),
            PType::F64 => Scalar::primitive(
                f64::from_bits(unordered_u64(
                    scalar
                        .as_primitive()
                        .typed_value::<u64>()
                        .vortex_expect("validated ordered float scalar"),
                )),
                array.dtype().nullability(),
            ),
            ptype => vortex_panic!("unsupported OrderedFloat ptype {ptype}"),
        })
    }
}

impl ValidityChild<OrderedFloat> for OrderedFloat {
    fn validity_child(array: ArrayView<'_, OrderedFloat>) -> ArrayRef {
        array.encoded().clone()
    }
}

impl SliceReduce for OrderedFloat {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            OrderedFloat::try_new(array.encoded().slice(range)?, array.dtype().as_ptype())?
                .into_array(),
        ))
    }
}

static RULES: ParentRuleSet<OrderedFloat> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(OrderedFloat))]);

pub trait OrderedFloatArrayExt: TypedArrayRef<OrderedFloat> + OrderedFloatArraySlotsExt {}

impl<T: TypedArrayRef<OrderedFloat>> OrderedFloatArrayExt for T {}

impl OrderedFloat {
    /// Estimate BlockResidual bytes for ordered float bits without materialized integer values.
    pub fn estimate_block_residual(
        array: ArrayView<'_, Primitive>,
    ) -> VortexResult<BlockResidualEstimate> {
        let BlockResidualCodecEstimate {
            encoded_nbytes,
            patch_count,
        } = match array.ptype() {
            PType::F16 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<f16>(), |value| {
                    u64::from(ordered_u16(value.to_bits()))
                })
            }
            PType::F32 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<f32>(), |value| {
                    u64::from(ordered_u32(value.to_bits()))
                })
            }
            PType::F64 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<f64>(), |value| {
                    ordered_u64(value.to_bits())
                })
            }
            ptype => vortex_bail!("OrderedFloat requires f16, f32, or f64, got {ptype}"),
        };
        let validity_nbytes = validity_to_child(&array.validity()?, array.len())
            .map(|validity| validity.nbytes())
            .unwrap_or(0);
        BlockResidualEstimate::try_new(encoded_nbytes, validity_nbytes, patch_count)
    }

    /// Construct an ordered float array from an unsigned child.
    pub fn try_new(encoded: ArrayRef, float_ptype: PType) -> VortexResult<OrderedFloatArray> {
        vortex_ensure!(
            matches!(float_ptype, PType::F16 | PType::F32 | PType::F64),
            "OrderedFloat requires f16, f32, or f64"
        );
        let dtype = DType::Primitive(float_ptype, encoded.dtype().nullability());
        let len = encoded.len();
        Array::try_from_parts(
            ArrayParts::new(OrderedFloat, dtype, len, OrderedFloatData)
                .with_slots(OrderedFloatSlots { encoded }.into_slots()),
        )
    }

    /// Map canonical floats to ordered unsigned integer bits.
    pub fn from_primitive(array: ArrayView<'_, Primitive>) -> VortexResult<OrderedFloatArray> {
        let validity = array.validity()?;
        match array.ptype() {
            PType::F16 => Self::try_new(
                PrimitiveArray::new(
                    Buffer::from(
                        array
                            .as_slice::<f16>()
                            .iter()
                            .map(|value| ordered_u16(value.to_bits()))
                            .collect::<Vec<_>>(),
                    ),
                    validity,
                )
                .into_array(),
                PType::F16,
            ),
            PType::F32 => Self::try_new(
                PrimitiveArray::new(
                    Buffer::from(
                        array
                            .as_slice::<f32>()
                            .iter()
                            .map(|value| ordered_u32(value.to_bits()))
                            .collect::<Vec<_>>(),
                    ),
                    validity,
                )
                .into_array(),
                PType::F32,
            ),
            PType::F64 => Self::try_new(
                PrimitiveArray::new(
                    Buffer::from(
                        array
                            .as_slice::<f64>()
                            .iter()
                            .map(|value| ordered_u64(value.to_bits()))
                            .collect::<Vec<_>>(),
                    ),
                    validity,
                )
                .into_array(),
                PType::F64,
            ),
            ptype => vortex_bail!("OrderedFloat requires f16, f32, or f64, got {ptype}"),
        }
    }
}

fn decode_primitive(
    array: ArrayView<'_, OrderedFloat>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = encoded.validity()?;
    Ok(match array.dtype().as_ptype() {
        PType::F16 => PrimitiveArray::new(
            encoded
                .into_buffer::<u16>()
                .map_each_in_place(|value| f16::from_bits(unordered_u16(value)))
                .freeze(),
            validity,
        ),
        PType::F32 => PrimitiveArray::new(
            encoded
                .into_buffer::<u32>()
                .map_each_in_place(|value| f32::from_bits(unordered_u32(value)))
                .freeze(),
            validity,
        ),
        PType::F64 => PrimitiveArray::new(
            encoded
                .into_buffer::<u64>()
                .map_each_in_place(|value| f64::from_bits(unordered_u64(value)))
                .freeze(),
            validity,
        ),
        ptype => vortex_panic!("unsupported OrderedFloat ptype {ptype}"),
    })
}

fn ordered_ptype(ptype: PType) -> VortexResult<PType> {
    match ptype {
        PType::F16 => Ok(PType::U16),
        PType::F32 => Ok(PType::U32),
        PType::F64 => Ok(PType::U64),
        _ => vortex_bail!("OrderedFloat requires f16, f32, or f64, got {ptype}"),
    }
}

fn ordered_u16(bits: u16) -> u16 {
    if bits & (1_u16 << 15) == 0 {
        bits ^ (1_u16 << 15)
    } else {
        !bits
    }
}

fn unordered_u16(value: u16) -> u16 {
    if value & (1_u16 << 15) == 0 {
        !value
    } else {
        value ^ (1_u16 << 15)
    }
}

fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn unordered_u32(value: u32) -> u32 {
    if value & (1_u32 << 31) == 0 {
        !value
    } else {
        value ^ (1_u32 << 31)
    }
}

fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn unordered_u64(value: u64) -> u64 {
    if value & (1_u64 << 63) == 0 {
        !value
    } else {
        value ^ (1_u64 << 63)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::half::f16;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::OrderedFloat;
    use super::OrderedFloatArraySlotsExt;
    use crate::BlockResidual;
    use crate::BlockResidualArrayExt;

    #[test]
    fn roundtrip_f16_special_values() -> VortexResult<()> {
        let values = [
            f16::from_bits(0xfc00),
            f16::from_f32(-1.0),
            f16::NEG_ZERO,
            f16::ZERO,
            f16::from_f32(1.0),
            f16::INFINITY,
            f16::from_bits(0x7e42),
        ];
        let primitive = PrimitiveArray::from_iter(values);
        let estimate = OrderedFloat::estimate_block_residual(primitive.as_view())?;
        let ordered = OrderedFloat::from_primitive(primitive.as_view())?;
        let residuals = BlockResidual::from_primitive(ordered.encoded().as_::<Primitive>())?;
        assert_eq!(estimate.nbytes(), residuals.nbytes());
        assert_eq!(estimate.patch_count(), residuals.patch_positions().len());
        let encoded = OrderedFloat::try_new(residuals.into_array(), PType::F16)?;
        let session = array_session();
        crate::initialize(&session);
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;

        assert_eq!(
            decoded
                .as_slice::<f16>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.map(f16::to_bits)
        );
        Ok(())
    }

    #[test]
    fn roundtrip_special_values() -> VortexResult<()> {
        let primitive = PrimitiveArray::from_iter([
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ]);
        let estimate = OrderedFloat::estimate_block_residual(primitive.as_view())?;
        let encoded = OrderedFloat::from_primitive(primitive.as_view())?;
        let residuals = BlockResidual::from_primitive(encoded.encoded().as_::<Primitive>())?;
        assert_eq!(estimate.nbytes(), residuals.nbytes());
        assert_eq!(estimate.patch_count(), residuals.patch_positions().len());
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(encoded, primitive.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn roundtrip_f32_special_values() -> VortexResult<()> {
        let values = [
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0042),
        ];
        let primitive = PrimitiveArray::from_iter(values);
        let encoded = OrderedFloat::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;

        assert_eq!(
            decoded
                .as_slice::<f32>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.map(f32::to_bits)
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_logical_and_child_types() {
        let u32_child = PrimitiveArray::from_iter([0_u32, 1, 2]).into_array();
        assert!(OrderedFloat::try_new(u32_child.clone(), PType::F64).is_err());
        assert!(OrderedFloat::try_new(u32_child, PType::U32).is_err());

        let u64_child = PrimitiveArray::from_iter([0_u64, 1, 2]).into_array();
        assert!(OrderedFloat::try_new(u64_child, PType::F32).is_err());
    }

    #[test]
    fn roundtrip_f32_block_residual() -> VortexResult<()> {
        let values = (0..2_050)
            .scan(1_000.0_f32, |value, index| {
                *value += ((index * 7_919 % 101) as f32 - 50.0) * 0.0001;
                Some(*value)
            })
            .collect::<Vec<_>>();
        let primitive = PrimitiveArray::from_iter(values.clone());
        let estimate = OrderedFloat::estimate_block_residual(primitive.as_view())?;
        let ordered = OrderedFloat::from_primitive(primitive.as_view())?;
        let residuals = BlockResidual::from_primitive(ordered.encoded().as_::<Primitive>())?;
        assert_eq!(estimate.nbytes(), residuals.nbytes());
        assert_eq!(estimate.patch_count(), residuals.patch_positions().len());
        let encoded = OrderedFloat::try_new(residuals.into_array(), PType::F32)?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(encoded, primitive, &mut ctx);
        assert_eq!(
            encoded
                .execute_scalar(1_024, &mut ctx)?
                .as_primitive()
                .typed_value::<f32>(),
            Some(values[1_024])
        );
        assert_arrays_eq!(
            encoded.into_array().slice(1_023..1_026)?,
            primitive.into_array().slice(1_023..1_026)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn nullable_serialized_slice_roundtrip() -> VortexResult<()> {
        let primitive = PrimitiveArray::new(
            Buffer::from(vec![f64::NEG_INFINITY, -0.0, 0.0, 42.25, f64::INFINITY]),
            Validity::from_iter([true, false, true, true, true]),
        );
        let encoded = OrderedFloat::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        assert!(encoded.execute_scalar(1, &mut ctx)?.is_null());

        let sliced = encoded.into_array().slice(1..5)?;
        let expected = primitive.into_array().slice(1..5)?;
        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let serialized =
            sliced.serialize(&array_context, &session, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &session,
        )?;

        assert!(decoded.is::<OrderedFloat>());
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn conformance() -> VortexResult<()> {
        let primitive = PrimitiveArray::new(
            Buffer::from(vec![f32::NEG_INFINITY, -0.0, 0.0, 42.25, f32::INFINITY]),
            Validity::from_iter([true, false, true, true, true]),
        );
        let direct = OrderedFloat::from_primitive(primitive.as_view())?;
        let residuals = BlockResidual::from_primitive(direct.encoded().as_::<Primitive>())?;
        let fused = OrderedFloat::try_new(residuals.into_array(), PType::F32)?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        for array in [direct.into_array(), fused.into_array()] {
            test_array_consistency(&array, &mut ctx);
        }
        Ok(())
    }
}
