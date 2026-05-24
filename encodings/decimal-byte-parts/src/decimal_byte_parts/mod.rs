// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
pub(crate) mod compute;
mod rules;
mod slice;

use prost::Message as _;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::decimal_byte_parts::compute::kernel::PARENT_KERNELS;
use crate::decimal_byte_parts::rules::PARENT_RULES;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

impl ArrayHash for DecimalBytePartsData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for DecimalBytePartsData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

impl VTable for DecimalByteParts {
    type TypedArrayData = DecimalBytePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("expected decimal dtype, got {}", dtype)
        };
        let msp = slots[MSP_SLOT]
            .as_ref()
            .vortex_expect("DecimalBytePartsArray msp slot");
        DecimalBytePartsData::validate(msp, *decimal_dtype, dtype, len)?;
        for slot in &slots[NUM_FIXED_SLOTS..] {
            let lower = slot
                .as_ref()
                .vortex_expect("DecimalBytePartsArray lower part slot");
            validate_lower_part(lower, len)?;
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DecimalBytePartsArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("DecimalBytePartsArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DecimalBytesPartsMetadata {
                zeroth_child_ptype: PType::try_from(array.msp().dtype())? as i32,
                lower_part_count: u32::try_from(array.num_lower_parts())
                    .vortex_expect("lower part count fits in u32"),
            }
            .encode_to_vec(),
        ))
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
        let metadata = DecimalBytesPartsMetadata::decode(metadata)?;
        if dtype.as_decimal_opt().is_none() {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        }

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let msp = children.get(0, &encoded_dtype, len)?;

        let lower_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let mut slots: ArraySlots = smallvec![Some(msp)];
        for i in 0..metadata.lower_part_count as usize {
            slots.push(Some(children.get(
                NUM_FIXED_SLOTS + i,
                &lower_dtype,
                len,
            )?));
        }
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, DecimalBytePartsData)
                .with_slots(slots),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == MSP_SLOT {
            "msp".to_string()
        } else {
            format!("lower_part_{}", idx - NUM_FIXED_SLOTS)
        }
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        to_canonical_decimal(&array, ctx).map(ExecutionResult::done)
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

/// Slot holding the most significant part of the decimal values.
pub(super) const MSP_SLOT: usize = 0;
/// Number of fixed (non lower-part) slots. The MSP occupies slot 0 and lower parts follow.
pub(super) const NUM_FIXED_SLOTS: usize = 1;

/// This array encodes decimals as between 1 and 4 columns of primitive typed children.
///
/// The most significant part (`msp`) stores the most significant decimal bits and must be a signed
/// integer array; it is nullable iff the decimal is nullable. Decimals whose values fit in a signed
/// 64-bit integer are stored as a single `msp` part. Wider decimals (`i128`, `i256`) are split into
/// an `msp` part plus a sequence of unsigned 64-bit `lower_parts`, in most-significant-first order.
/// The whole-array validity always lives in the `msp`; lower parts are non-nullable.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower_part\[0\] = 64..0.
/// For a decimal i256 the msp holds bits \[255..192\] and lower_part\[0..3\] hold the three remaining
/// 64-bit chunks in descending significance.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsData;

impl Display for DecimalBytePartsData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct DecimalBytePartsDataParts {
    pub msp: ArrayRef,
}

pub trait DecimalBytePartsArrayExt: TypedArrayRef<DecimalByteParts> {
    /// The most significant part array, carrying the array validity.
    fn msp(&self) -> &ArrayRef {
        self.as_ref().slots()[MSP_SLOT]
            .as_ref()
            .vortex_expect("DecimalBytePartsArray msp slot")
    }

    /// The number of lower (least significant) 64-bit parts.
    fn num_lower_parts(&self) -> usize {
        self.as_ref().slots().len() - NUM_FIXED_SLOTS
    }

    /// The lower part at the given index, in most-significant-first order.
    fn lower_part(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[NUM_FIXED_SLOTS + idx]
            .as_ref()
            .vortex_expect("DecimalBytePartsArray lower part slot")
    }

    /// The lower parts, in most-significant-first order.
    fn lower_parts(&self) -> Vec<ArrayRef> {
        (0..self.num_lower_parts())
            .map(|idx| self.lower_part(idx).clone())
            .collect()
    }
}

impl<T: TypedArrayRef<DecimalByteParts>> DecimalBytePartsArrayExt for T {}

impl DecimalBytePartsData {
    pub fn validate(
        msp: &ArrayRef,
        decimal_dtype: DecimalDType,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        let expected_dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        vortex_ensure!(
            dtype == &expected_dtype,
            "expected dtype {expected_dtype}, got {dtype}"
        );
        vortex_ensure!(msp.len() == len, "expected len {len}, got {}", msp.len());
        Ok(())
    }
}

/// Validates a single lower part: it must be a non-nullable unsigned integer array matching the
/// array length.
fn validate_lower_part(lower: &ArrayRef, len: usize) -> VortexResult<()> {
    vortex_ensure!(
        lower.dtype().is_unsigned_int(),
        "decimal byte parts lower part must be an unsigned integer array, got {}",
        lower.dtype()
    );
    vortex_ensure!(
        !lower.dtype().is_nullable(),
        "decimal byte parts lower part must be non-nullable, got {}",
        lower.dtype()
    );
    vortex_ensure!(
        lower.len() == len,
        "decimal byte parts lower part length {} does not match {len}",
        lower.len()
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct DecimalByteParts;

impl DecimalByteParts {
    /// Construct a new single-part [`DecimalBytePartsArray`] from an MSP array and decimal dtype.
    ///
    /// The decimal values must fit entirely within the signed `msp` integer width.
    pub fn try_new(
        msp: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        Self::try_new_parts(msp, Vec::new(), decimal_dtype)
    }

    /// Construct a new [`DecimalBytePartsArray`] from an MSP array, a sequence of unsigned 64-bit
    /// lower parts (most-significant-first), and the decimal dtype.
    ///
    /// The reconstructed decimal value for row `i` is
    /// `(msp[i] << (64 * num_lower)) | (lower[0][i] << (64 * (num_lower - 1))) | .. | lower[last][i]`.
    pub fn try_new_parts(
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        let mut slots: ArraySlots = smallvec![Some(msp)];
        slots.extend(lower_parts.into_iter().map(Some));
        Array::try_from_parts(
            ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData).with_slots(slots),
        )
    }
}

/// Converts a DecimalBytePartsArray to its canonical DecimalArray representation.
fn to_canonical_decimal(
    array: &DecimalBytePartsArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decimal_dtype = *array
        .dtype()
        .as_decimal_opt()
        .vortex_expect("must be a decimal dtype");
    let msp = array.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = msp.validity()?;
    let len = msp.len();

    match array.num_lower_parts() {
        // Single part: the MSP buffer is the decimal buffer directly.
        0 => Ok(match_each_signed_integer_ptype!(msp.ptype(), |P| {
            // SAFETY: The primitive array's buffer is already validated with correct type.
            // The decimal dtype matches the array's dtype, and validity is preserved.
            unsafe { DecimalArray::new_unchecked(msp.to_buffer::<P>(), decimal_dtype, validity) }
                .into_array()
        })),
        // i128: msp holds the high 64 bits, lower[0] the low 64 bits.
        1 => {
            let msp_hi = msp_as_i128(&msp);
            let lower = array.lower_part(0).clone().execute::<PrimitiveArray>(ctx)?;
            let lower = lower.as_slice::<u64>();
            let mut buffer = BufferMut::<i128>::with_capacity(len);
            for (hi, lo) in msp_hi.iter().zip(lower.iter()) {
                buffer.push((hi << 64) | i128::from(*lo));
            }
            // SAFETY: reconstructed values match the decimal dtype, validity comes from the msp.
            Ok(
                unsafe { DecimalArray::new_unchecked(buffer.freeze(), decimal_dtype, validity) }
                    .into_array(),
            )
        }
        // i256: msp holds bits [255..192], lower parts the three remaining 64-bit chunks.
        3 => {
            let msp_hi = msp_as_i128(&msp);
            let p0 = array.lower_part(0).clone().execute::<PrimitiveArray>(ctx)?;
            let p1 = array.lower_part(1).clone().execute::<PrimitiveArray>(ctx)?;
            let p2 = array.lower_part(2).clone().execute::<PrimitiveArray>(ctx)?;
            let (p0, p1, p2) = (
                p0.as_slice::<u64>(),
                p1.as_slice::<u64>(),
                p2.as_slice::<u64>(),
            );
            let mut buffer = BufferMut::<i256>::with_capacity(len);
            for i in 0..len {
                let upper = (msp_hi[i] << 64) | i128::from(p0[i]);
                let lower = (u128::from(p1[i]) << 64) | u128::from(p2[i]);
                buffer.push(i256::from_parts(lower, upper));
            }
            // SAFETY: reconstructed values match the decimal dtype, validity comes from the msp.
            Ok(
                unsafe { DecimalArray::new_unchecked(buffer.freeze(), decimal_dtype, validity) }
                    .into_array(),
            )
        }
        n => vortex_bail!("unsupported decimal byte parts lower part count {n}"),
    }
}

/// Reads the signed MSP primitive array, widening every element to `i128`.
fn msp_as_i128(msp: &PrimitiveArray) -> Vec<i128> {
    match_each_signed_integer_ptype!(msp.ptype(), |P| {
        msp.as_slice::<P>().iter().map(|v| i128::from(*v)).collect()
    })
}

/// Reads the unsigned `u64` value of the `idx`-th lower part at `index`.
fn lower_part_at(
    array: &ArrayView<'_, DecimalByteParts>,
    idx: usize,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    Ok(array
        .lower_part(idx)
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("non-null"))
}

impl OperationsVTable<DecimalByteParts> for DecimalByteParts {
    fn scalar_at(
        array: ArrayView<'_, DecimalByteParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Null positions are handled by the validity vtable before reaching here, so the msp value
        // and all lower part values at `index` are non-null.
        let msp = array
            .msp()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-null");

        let value = match array.num_lower_parts() {
            0 => DecimalValue::I64(msp),
            1 => {
                let lo = lower_part_at(&array, 0, index, ctx)?;
                DecimalValue::I128((i128::from(msp) << 64) | i128::from(lo))
            }
            3 => {
                let p0 = lower_part_at(&array, 0, index, ctx)?;
                let p1 = lower_part_at(&array, 1, index, ctx)?;
                let p2 = lower_part_at(&array, 2, index, ctx)?;
                let upper = (i128::from(msp) << 64) | i128::from(p0);
                let lower = (u128::from(p1) << 64) | u128::from(p2);
                DecimalValue::I256(i256::from_parts(lower, upper))
            }
            n => vortex_bail!("unsupported decimal byte parts lower part count {n}"),
        };

        Scalar::try_new(array.dtype().clone(), Some(ScalarValue::Decimal(value)))
    }
}

impl ValidityChild<DecimalByteParts> for DecimalByteParts {
    fn validity_child(array: ArrayView<'_, DecimalByteParts>) -> ArrayRef {
        // validity stored in 0th child
        array.msp().clone()
    }
}

#[cfg(test)]
// Splitting decimal values into 64-bit parts intentionally truncates wider integers.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::DecimalByteParts;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(DecimalByteParts);
        session
    });

    /// Serializes the array and decodes it back, returning the round-tripped array.
    fn serde_round_trip(array: vortex_array::ArrayRef) -> VortexResult<vortex_array::ArrayRef> {
        let context = ArrayContext::empty();
        let bytes = array
            .serialize(
                &context,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )?
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<vortex_buffer::BufferMut<u8>>()
            .freeze();
        SerializedArray::try_from(bytes)?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(context.to_ids()),
            &SESSION,
        )
    }

    #[test]
    fn test_scalar_at_decimal_parts() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let array = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200i32, 400i32],
                Validity::Array(BoolArray::from_iter(vec![false, true, true]).into_array()),
            )
            .into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();

        assert_eq!(
            Scalar::null(dtype.clone()),
            array
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(
                dtype.clone(),
                Some(ScalarValue::Decimal(DecimalValue::I64(200)))
            )
            .unwrap(),
            array
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I64(400)))).unwrap(),
            array
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
    }

    /// Builds a two-part (`i128`) byte-parts array from raw values and the bit-split the compressor
    /// uses, then checks it canonicalizes back to the equivalent `DecimalArray`.
    fn assert_i128_round_trip(values: &[i128], decimal_dtype: DecimalDType) -> VortexResult<()> {
        let msp = PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
        let low = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
        let array = DecimalByteParts::try_new_parts(msp, vec![low], decimal_dtype)?.into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = array.execute::<DecimalArray>(&mut ctx)?;
        let expected = DecimalArray::from_iter(values.iter().copied(), decimal_dtype);
        assert_arrays_eq!(canonical.into_array(), expected.into_array());
        Ok(())
    }

    /// Builds a four-part (`i256`) byte-parts array and checks it canonicalizes back.
    fn assert_i256_round_trip(values: &[i256], decimal_dtype: DecimalDType) -> VortexResult<()> {
        let msp = PrimitiveArray::from_iter(values.iter().map(|v| {
            let (_, upper) = v.to_parts();
            (upper >> 64) as i64
        }))
        .into_array();
        let p0 =
            PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().1 as u64)).into_array();
        let p1 = PrimitiveArray::from_iter(values.iter().map(|v| (v.to_parts().0 >> 64) as u64))
            .into_array();
        let p2 =
            PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().0 as u64)).into_array();
        let array =
            DecimalByteParts::try_new_parts(msp, vec![p0, p1, p2], decimal_dtype)?.into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = array.execute::<DecimalArray>(&mut ctx)?;
        let expected = DecimalArray::from_iter(values.iter().copied(), decimal_dtype);
        assert_arrays_eq!(canonical.into_array(), expected.into_array());
        Ok(())
    }

    #[test]
    fn test_i128_round_trip() -> VortexResult<()> {
        let values = vec![
            0i128,
            1,
            -1,
            i64::MAX as i128 + 1,
            -(i64::MAX as i128) - 10,
            10i128.pow(30),
            -(10i128.pow(30)),
            10i128.pow(37),
            -(10i128.pow(37)),
        ];
        assert_i128_round_trip(&values, DecimalDType::new(38, 4))
    }

    #[test]
    fn test_i256_round_trip() -> VortexResult<()> {
        let values = vec![
            i256::ZERO,
            i256::from_i128(1),
            i256::from_i128(-1),
            i256::from_i128(i128::MAX),
            i256::from_i128(i128::MIN),
            i256::from_parts(u128::MAX, 1),
            i256::from_parts(0, -1),
            i256::from_parts(12345678901234567890, 98765432109876543210i128),
        ];
        assert_i256_round_trip(&values, DecimalDType::new(76, 8))
    }

    #[test]
    fn test_i128_round_trip_nullable() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(38, 2);
        let raw = [
            Some(10i128.pow(30)),
            None,
            Some(-(10i128.pow(25))),
            Some(0),
            None,
        ];
        let validity =
            Validity::Array(BoolArray::from_iter(raw.iter().map(|v| v.is_some())).into_array());
        let msp = PrimitiveArray::new(
            raw.iter()
                .map(|v| (v.unwrap_or(0) >> 64) as i64)
                .collect::<vortex_buffer::Buffer<i64>>(),
            validity,
        )
        .into_array();
        let low = PrimitiveArray::from_iter(raw.iter().map(|v| v.unwrap_or(0) as u64)).into_array();
        let array = DecimalByteParts::try_new_parts(msp, vec![low], decimal_dtype)?.into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = array.execute::<DecimalArray>(&mut ctx)?;
        let expected = DecimalArray::from_option_iter(raw.iter().copied(), decimal_dtype);
        assert_arrays_eq!(canonical.into_array(), expected.into_array());
        Ok(())
    }

    #[test]
    fn test_scalar_at_multi_part() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(38, 0);
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let value = 10i128.pow(30) + 7;
        let msp = PrimitiveArray::from_iter([(value >> 64) as i64]).into_array();
        let low = PrimitiveArray::from_iter([value as u64]).into_array();
        let array = DecimalByteParts::try_new_parts(msp, vec![low], decimal_dtype)?.into_array();

        let scalar = array.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(
            Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I128(value))))?,
            scalar
        );
        Ok(())
    }

    #[test]
    fn test_serde_round_trip_i128() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(38, 4);
        let values: Vec<i128> = vec![0, 1, -1, 10i128.pow(30), -(10i128.pow(30)), 10i128.pow(37)];
        let msp = PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
        let low = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
        let array = DecimalByteParts::try_new_parts(msp, vec![low], decimal_dtype)?.into_array();

        let decoded = serde_round_trip(array.clone())?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_arrays_eq!(
            decoded.execute::<DecimalArray>(&mut ctx)?.into_array(),
            array.execute::<DecimalArray>(&mut ctx)?.into_array()
        );
        Ok(())
    }

    #[test]
    fn test_serde_round_trip_i256() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(76, 8);
        let values: Vec<i256> = vec![
            i256::ZERO,
            i256::from_i128(-1),
            i256::from_parts(u128::MAX, 1),
            i256::from_parts(12345678901234567890, 98765432109876543210i128),
        ];
        let msp = PrimitiveArray::from_iter(values.iter().map(|v| (v.to_parts().1 >> 64) as i64))
            .into_array();
        let p0 =
            PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().1 as u64)).into_array();
        let p1 = PrimitiveArray::from_iter(values.iter().map(|v| (v.to_parts().0 >> 64) as u64))
            .into_array();
        let p2 =
            PrimitiveArray::from_iter(values.iter().map(|v| v.to_parts().0 as u64)).into_array();
        let array =
            DecimalByteParts::try_new_parts(msp, vec![p0, p1, p2], decimal_dtype)?.into_array();

        let decoded = serde_round_trip(array.clone())?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_arrays_eq!(
            decoded.execute::<DecimalArray>(&mut ctx)?.into_array(),
            array.execute::<DecimalArray>(&mut ctx)?.into_array()
        );
        Ok(())
    }
}
