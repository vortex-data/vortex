// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayView;
pub(crate) mod compute;
mod limbs;
mod rules;
mod slice;
#[cfg(test)]
pub(crate) mod testing;

pub use limbs::DecimalParts;
pub use limbs::MAX_LOWER_PARTS;
pub use limbs::split_decimal;
use prost::Message as _;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::PType;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::decimal_byte_parts::limbs::LOWER_PART_DTYPE;
use crate::decimal_byte_parts::limbs::assemble_decimal;
use crate::decimal_byte_parts::limbs::assembled_values_type;
use crate::decimal_byte_parts::limbs::combine_i128;
use crate::decimal_byte_parts::limbs::combine_i256;
use crate::decimal_byte_parts::rules::PARENT_RULES;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

impl ArrayHash for DecimalBytePartsData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for DecimalBytePartsData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
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

impl DecimalBytesPartsMetadata {
    /// The number of lower parts encoded in this array.
    ///
    /// # Errors
    ///
    /// Returns an error if the count exceeds [`MAX_LOWER_PARTS`].
    fn lower_part_count(&self) -> VortexResult<usize> {
        let count = usize::try_from(self.lower_part_count)
            .map_err(|_| vortex_err!("lower part count {} out of range", self.lower_part_count))?;
        vortex_ensure!(
            count <= MAX_LOWER_PARTS,
            "at most {MAX_LOWER_PARTS} lower parts are supported, got {count}"
        );
        Ok(count)
    }
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
        let slots = DecimalBytePartsSlotsView::from_slots(slots);
        DecimalBytePartsData::validate(
            slots.msp,
            slots.lower_parts.iter(),
            *decimal_dtype,
            dtype,
            len,
        )
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let lower_part_count = u32::try_from(array.lower_parts().len())
            .map_err(|_| vortex_err!("lower part count exceeds u32"))?;
        Ok(Some(
            DecimalBytesPartsMetadata {
                zeroth_child_ptype: PType::try_from(array.msp().dtype())? as i32,
                lower_part_count,
            }
            .encode_to_vec(),
        ))
    }

    fn serialized_id(&self, array: ArrayView<'_, Self>) -> ArrayId {
        // The frozen `vortex.decimal_byte_parts` format promises a single child: readers that
        // predate lower parts require `lower_part_count == 0`, so an array carrying lower
        // parts must serialize under the v2 format id instead. That id is what the writer's
        // permitted-encoding check gates, and what a reader without v2 support rejects as an
        // unknown encoding instead of misreading the children.
        if array.lower_parts().is_empty() {
            VTable::id(self)
        } else {
            ArrayPlugin::id(&DecimalBytePartsV2)
        }
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
        vortex_ensure!(
            dtype.as_decimal_opt().is_some(),
            "decoding decimal but given non decimal dtype {dtype}"
        );

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let lower_part_count = metadata.lower_part_count()?;
        vortex_ensure!(
            children.len() == DecimalBytePartsSlots::FIXED_COUNT + lower_part_count,
            "expected {} children, got {}",
            DecimalBytePartsSlots::FIXED_COUNT + lower_part_count,
            children.len()
        );

        let msp = children.get(DecimalBytePartsSlots::MSP, &encoded_dtype, len)?;

        let mut slots = ArraySlots::with_capacity(children.len());
        slots.push(Some(msp));
        for idx in 0..lower_part_count {
            slots.push(Some(children.get(
                DecimalBytePartsSlots::LOWER_PARTS_OFFSET + idx,
                &LOWER_PART_DTYPE,
                len,
            )?));
        }

        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, DecimalBytePartsData)
                .with_slots(slots),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DecimalBytePartsSlots::slot_name(idx)
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
}

#[array_slots(DecimalByteParts)]
pub struct DecimalBytePartsSlots {
    /// The most significant parts of the decimal values.
    #[slot(0)]
    pub msp: ArrayRef,
    /// The remaining 64-bit windows of the decimal values, most significant first.
    #[slot(1..)]
    pub lower_parts: Vec<ArrayRef>,
}

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// The most significant part (msp) storing the most significant decimal bits.
/// This array must be signed and is nullable iff the decimal is nullable.
/// Every lower part is a non-nullable `u64` holding a raw 64-bit window of the value.
///
/// e.g. for a decimal i128 \[ 127..64 | 63..0 \] msp = 127..64 and lower_part\[0\] = 63..0
///
/// All parts live in slots, so the array carries no additional data.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsData;

impl Display for DecimalBytePartsData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl DecimalBytePartsData {
    /// Validate the parts of a [`DecimalBytePartsArray`].
    ///
    /// # Errors
    ///
    /// Returns an error if the MSP is not a signed integer array of length `len`, if `dtype`
    /// does not match the MSP's nullability, or if any lower part is not a non-nullable
    /// `u64` array of length `len`.
    pub fn validate<'a>(
        msp: &ArrayRef,
        lower_parts: impl ExactSizeIterator<Item = &'a ArrayRef>,
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

        let lower_part_count = lower_parts.len();
        for (idx, part) in lower_parts.enumerate() {
            vortex_ensure!(
                part.dtype() == &LOWER_PART_DTYPE,
                "lower part {idx} must have dtype {LOWER_PART_DTYPE}, got {}",
                part.dtype()
            );
            vortex_ensure!(
                part.len() == len,
                "lower part {idx} has len {}, expected {len}",
                part.len()
            );
        }
        // Rejects part combinations that cannot be reassembled into a decimal value. This also
        // bounds the lower part count.
        let values_type = assembled_values_type(msp.dtype().as_ptype(), lower_part_count)?;

        // The parts must not assemble into a wider value than the declared precision holds.
        // Without this, a crafted array carrying more parts than its precision needs
        // canonicalizes to out-of-precision values that then panic in the scalar path.
        let widest = DecimalType::smallest_decimal_value_type(&decimal_dtype);
        vortex_ensure!(
            values_type <= widest,
            "parts assemble into {values_type:?}, wider than the {widest:?} required by \
             decimal precision {}",
            decimal_dtype.precision()
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DecimalByteParts;

impl DecimalByteParts {
    /// Construct a new [`DecimalBytePartsArray`] from an MSP array and decimal dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the MSP is not a signed integer array.
    pub fn try_new(
        msp: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        Self::try_new_with_lower_parts(msp, Vec::new(), decimal_dtype)
    }

    /// Construct a new [`DecimalBytePartsArray`] from an MSP array, its lower parts, and a
    /// decimal dtype.
    ///
    /// Lower parts are ordered most significant first and must each be a non-nullable `u64`
    /// array of the same length as the MSP. See [`split_decimal`] for producing them from a
    /// canonical decimal array.
    ///
    /// # Errors
    ///
    /// Returns an error if the parts do not describe a valid decimal, see
    /// [`DecimalBytePartsData::validate`].
    pub fn try_new_with_lower_parts(
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        // Building lower parts in memory is never gated — reading a file requires it. What is
        // gated is the serialized form: an array carrying lower parts serializes under the
        // `DecimalBytePartsV2` format id, which only editions that contain it may write.
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        let slots = DecimalBytePartsSlots { msp, lower_parts }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData).with_slots(slots),
        )
    }
}

/// The `vortex.decimal_byte_parts_v2` serialized format: byte parts carrying lower parts.
///
/// This is a serialized format, not a second in-memory encoding. `vortex.decimal_byte_parts`
/// froze promising a single child, so an array with lower parts serializes under this id
/// instead — see [`VTable::serialized_id`] on [`DecimalByteParts`] — and both ids deserialize
/// back into the same [`DecimalBytePartsArray`]. A reader that predates lower parts fails on
/// this id with an unknown-encoding error rather than misreading the children.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsV2;

impl ArrayPlugin for DecimalBytePartsV2 {
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal_byte_parts_v2");
        *ID
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // In-memory arrays always carry the `DecimalByteParts` encoding id, so metadata
        // serialization is resolved through that plugin; both formats share it.
        ArrayPlugin::serialize(&DecimalByteParts, array, session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        Ok(Array::try_from_parts(VTable::deserialize(
            &DecimalByteParts,
            dtype,
            len,
            metadata,
            buffers,
            children,
            session,
        )?)?
        .into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        *id == ArrayPlugin::id(self) || *id == VTable::id(&DecimalByteParts)
    }
}

/// The decimal storage type this array canonicalizes to.
fn values_type(array: ArrayView<'_, DecimalByteParts>) -> VortexResult<DecimalType> {
    assembled_values_type(array.msp().dtype().as_ptype(), array.lower_parts().len())
}

/// The decimal dtype this array carries.
///
/// Guaranteed to be a decimal by construction: [`DecimalBytePartsData::validate`] rejects
/// every other dtype.
pub(crate) fn decimal_dtype(array: ArrayView<'_, DecimalByteParts>) -> DecimalDType {
    *array
        .dtype()
        .as_decimal_opt()
        .vortex_expect("must be a decimal dtype")
}

/// Rebuild the array by applying `f` to the MSP and to every lower part, in slot order.
///
/// Part-wise operations must touch every part. Going through this rather than calling
/// [`DecimalByteParts::try_new_with_lower_parts`] directly makes dropping a lower part —
/// which silently corrupts wide values — unrepresentable.
pub(crate) fn map_parts(
    array: ArrayView<'_, DecimalByteParts>,
    mut f: impl FnMut(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<DecimalBytePartsArray> {
    let msp = f(array.msp())?;
    let lower_parts = array
        .lower_parts()
        .iter()
        .map(&mut f)
        .collect::<VortexResult<Vec<_>>>()?;
    DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, decimal_dtype(array))
}

/// Rebuild the array with a replacement MSP, keeping its lower parts untouched.
///
/// Only valid for operations that cannot change a row's magnitude bits — a nullability cast
/// or a mask — since the lower parts keep whatever bits they held. That is sound because
/// validity lives in the MSP alone, so lower-part bits in a null row are already undefined.
pub(crate) fn with_msp(
    array: ArrayView<'_, DecimalByteParts>,
    msp: ArrayRef,
    decimal_dtype: DecimalDType,
) -> VortexResult<DecimalBytePartsArray> {
    DecimalByteParts::try_new_with_lower_parts(msp, array.lower_parts().to_vec(), decimal_dtype)
}

/// Converts a DecimalBytePartsArray to its canonical DecimalArray representation.
fn to_canonical_decimal(
    array: &DecimalBytePartsArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let msp = array.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let lower_parts = array
        .lower_parts()
        .iter()
        .map(|part| part.clone().execute::<PrimitiveArray>(ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(assemble_decimal(&msp, &lower_parts, decimal_dtype(array.as_view()))?.into_array())
}

impl OperationsVTable<DecimalByteParts> for DecimalByteParts {
    fn scalar_at(
        array: ArrayView<'_, DecimalByteParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let scalar = array.msp().execute_scalar(index, ctx)?;

        // Note. values in msp, can only be signed integers upto size i64.
        let primitive_scalar = scalar.as_primitive();
        let msp = primitive_scalar.as_::<i64>().vortex_expect("non-null");

        let lower_parts = array
            .lower_parts()
            .iter()
            .map(|part| {
                Ok(part
                    .execute_scalar(index, ctx)?
                    .as_primitive()
                    .as_::<u64>()
                    .vortex_expect("lower parts are non-nullable"))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let value = if lower_parts.is_empty() {
            DecimalValue::I64(msp)
        } else {
            match values_type(array)? {
                DecimalType::I256 => DecimalValue::I256(combine_i256(msp, lower_parts.into_iter())),
                _ => DecimalValue::I128(combine_i128(msp, lower_parts)),
            }
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
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::DecimalByteParts;
    use crate::decimal_byte_parts::testing::encode;
    use crate::decimal_byte_parts::testing::i128_parts;
    use crate::decimal_byte_parts::testing::i256_of;
    use crate::decimal_byte_parts::testing::i256_parts;

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
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(
                dtype.clone(),
                Some(ScalarValue::Decimal(DecimalValue::I64(200)))
            )
            .unwrap(),
            array
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I64(400)))).unwrap(),
            array
                .execute_scalar(2, &mut array_session().create_execution_ctx())
                .unwrap()
        );
    }

    /// The largest unscaled value a `Decimal(38, _)` can hold: `10^38 - 1`.
    const MAX_PRECISION_38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

    /// The largest unscaled value a `Decimal(76, _)` can hold: `10^76 - 1`.
    fn max_precision_76() -> i256 {
        i256::from_i128(10).wrapping_pow(76) - i256::ONE
    }

    /// Values that exercise every 64-bit window of an `i128`, both signs, and the boundaries
    /// where a lower part carries into the MSP.
    fn wide_i128_values() -> Vec<i128> {
        vec![
            0,
            1,
            -1,
            (1 << 64) - 1,
            1 << 64,
            -(1 << 64),
            -((1 << 64) + 1),
            MAX_PRECISION_38,
            -MAX_PRECISION_38,
            1 << 100,
        ]
    }

    /// Values that exercise every 64-bit window of an `i256`.
    fn wide_i256_values() -> Vec<i256> {
        vec![
            i256::ZERO,
            i256::ONE,
            i256::ZERO - i256::ONE,
            i256_of(0, u128::MAX),
            i256_of(1, 0),
            i256_of(-1, 0),
            i256_of(-1, u128::MAX - 1),
            i256_of(1 << 64, 12345),
            max_precision_76(),
            i256::ZERO - max_precision_76(),
        ]
    }

    #[rstest]
    #[case::i128_non_nullable(i128_parts(wide_i128_values(), Validity::NonNullable))]
    #[case::i256_non_nullable(i256_parts(wide_i256_values(), Validity::NonNullable))]
    fn test_canonical_decimal_round_trips(
        #[case] array: DecimalBytePartsArray,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let canonical = array
            .clone()
            .into_array()
            .execute::<DecimalArray>(&mut ctx)?;
        assert_arrays_eq!(array, canonical, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_lower_part_layout_i128() -> VortexResult<()> {
        let array = i128_parts(vec![(3i128 << 64) | 7], Validity::NonNullable);
        assert_eq!(array.lower_parts().len(), 1);
        assert_eq!(array.msp().dtype().as_ptype(), PType::I64);
        assert_eq!(array.lower_parts()[0].dtype(), &LOWER_PART_DTYPE);

        let mut ctx = array_session().create_execution_ctx();
        let msp = array.msp().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let lower = array.lower_parts()[0]
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(msp.as_slice::<i64>(), &[3]);
        assert_eq!(lower.as_slice::<u64>(), &[7]);
        Ok(())
    }

    #[test]
    fn test_lower_part_layout_i256() -> VortexResult<()> {
        let array = i256_parts(
            vec![i256_of((5i128 << 64) | 6, (7u128 << 64) | 8)],
            Validity::NonNullable,
        );
        assert_eq!(array.lower_parts().len(), MAX_LOWER_PARTS);

        let mut ctx = array_session().create_execution_ctx();
        let msp = array.msp().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(msp.as_slice::<i64>(), &[5]);
        for (part, expected) in array.lower_parts().iter().zip([6u64, 7, 8]) {
            let part = part.clone().execute::<PrimitiveArray>(&mut ctx)?;
            assert_eq!(part.as_slice::<u64>(), &[expected]);
        }
        Ok(())
    }

    #[rstest]
    #[case::i128(i128_parts(wide_i128_values(), Validity::AllValid))]
    #[case::i256(i256_parts(wide_i256_values(), Validity::AllValid))]
    fn test_scalar_at_matches_canonical(#[case] array: DecimalBytePartsArray) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let canonical = array
            .clone()
            .into_array()
            .execute::<DecimalArray>(&mut ctx)?
            .into_array();
        let array = array.into_array();
        for idx in 0..array.len() {
            assert_eq!(
                array.execute_scalar(idx, &mut ctx)?,
                canonical.execute_scalar(idx, &mut ctx)?,
                "scalar mismatch at index {idx}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_scalar_at_null_with_lower_parts() -> VortexResult<()> {
        let array = i128_parts(
            vec![1i128 << 100, 2, 3],
            Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            array.execute_scalar(0, &mut ctx)?,
            Scalar::null(array.dtype().clone())
        );
        assert_eq!(
            array.execute_scalar(1, &mut ctx)?,
            Scalar::decimal(
                DecimalValue::I128(2),
                DecimalDType::new(38, 2),
                Nullability::Nullable
            )
        );
        Ok(())
    }

    #[rstest]
    #[case::one_lower_part(i128_parts(wide_i128_values(), Validity::NonNullable))]
    #[case::three_lower_parts(i256_parts(wide_i256_values(), Validity::NonNullable))]
    #[case::nullable_three_lower_parts(i256_parts(wide_i256_values(), Validity::AllValid))]
    fn test_serde_round_trip_with_lower_parts(
        #[case] array: DecimalBytePartsArray,
    ) -> VortexResult<()> {
        test_serde_round_trip(array)
    }

    #[rstest]
    #[case::no_lower_parts(
        encode(&DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(9, 2), Validity::NonNullable))
            .vortex_expect("valid decimal byte parts")
    )]
    fn test_serde_round_trip_flat(#[case] array: DecimalBytePartsArray) -> VortexResult<()> {
        test_serde_round_trip(array)
    }

    fn test_serde_round_trip(array: DecimalBytePartsArray) -> VortexResult<()> {
        let session = array_session();
        // Both serialized formats must be registered: an array with lower parts comes back
        // under the v2 format id.
        crate::initialize(&session);

        let array = array.into_array();
        let dtype = array.dtype().clone();
        let len = array.len();
        let lower_part_count = array
            .as_opt::<DecimalByteParts>()
            .vortex_expect("byte parts array")
            .lower_parts()
            .len();

        let array_ctx = ArrayContext::empty();
        let serialized = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

        assert_eq!(
            decoded
                .as_opt::<DecimalByteParts>()
                .vortex_expect("byte parts array")
                .lower_parts()
                .len(),
            lower_part_count,
            "lower parts must survive serde"
        );

        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(array, decoded, &mut ctx);
        Ok(())
    }

    fn msp() -> ArrayRef {
        buffer![1i64, 2, 3].into_array()
    }

    fn lower_part() -> ArrayRef {
        buffer![1u64, 2, 3].into_array()
    }

    #[rstest]
    #[case::signed_lower_part(vec![buffer![1i64, 2, 3].into_array()], DecimalDType::new(38, 2))]
    #[case::nullable_lower_part(
        vec![PrimitiveArray::new(buffer![1u64, 2, 3], Validity::AllValid).into_array()],
        DecimalDType::new(38, 2)
    )]
    #[case::mismatched_length(vec![buffer![1u64, 2].into_array()], DecimalDType::new(38, 2))]
    #[case::too_many_parts(
        vec![lower_part(), lower_part(), lower_part(), lower_part()],
        DecimalDType::new(76, 2)
    )]
    // Parts assembling into an i256 under a precision that only needs i128 would canonicalize
    // to values outside the declared precision.
    #[case::wider_than_precision(vec![lower_part(), lower_part()], DecimalDType::new(38, 2))]
    fn test_rejects_invalid_parts(
        #[case] lower_parts: Vec<ArrayRef>,
        #[case] decimal_dtype: DecimalDType,
    ) {
        assert!(
            DecimalByteParts::try_new_with_lower_parts(msp(), lower_parts, decimal_dtype).is_err()
        );
    }

    fn deserialize_with(
        lower_part_count: u32,
        children: Vec<ArrayRef>,
    ) -> VortexResult<ArrayParts<DecimalByteParts>> {
        let metadata = DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::I64 as i32,
            lower_part_count,
        };
        VTable::deserialize(
            &DecimalByteParts,
            &DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
            3,
            &metadata.encode_to_vec(),
            &[],
            &children,
            &array_session(),
        )
    }

    #[test]
    fn test_deserialize_reads_lower_parts() -> VortexResult<()> {
        let parts = deserialize_with(1, vec![msp(), lower_part()])?;
        let array = Array::try_from_parts(parts)?;
        assert_eq!(array.lower_parts().len(), 1);

        let mut ctx = array_session().create_execution_ctx();
        let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(
            canonical.buffer::<i128>().as_slice(),
            &[(1i128 << 64) | 1, (2i128 << 64) | 2, (3i128 << 64) | 3]
        );
        Ok(())
    }

    /// An array read from a file can be handed straight back to a writer, bypassing both the
    /// constructor and the compressor. Its serialized id must still be the v2 format, so a
    /// writer whose permitted encodings predate the v2 format refuses it.
    #[test]
    fn read_lower_parts_serialize_under_the_wide_format() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);

        let array = deserialize_with(1, vec![msp(), lower_part()])
            .and_then(Array::try_from_parts)?
            .into_array();

        assert_eq!(
            session.array_serialized_id(&array)?,
            ArrayPlugin::id(&DecimalBytePartsV2)
        );

        let restricted = ArrayContext::empty()
            .with_allowed_ids([VTable::id(&DecimalByteParts)].into_iter().collect());
        let err = array
            .serialize(&restricted, &session, &SerializeOptions::default())
            .expect_err("expected the permitted-encoding check to refuse the v2 format");
        assert!(
            err.to_string().contains("not permitted"),
            "error should name the permitted-encoding check, got: {err}"
        );
        Ok(())
    }

    /// Reading back an array that already carries lower parts, and computing over it, must
    /// always work: the v2 format only restricts which writers may emit it. If reading or
    /// the rebuild that every compute kernel does were blocked, a session whose editions
    /// predate the v2 format could not read a file written by one that includes it.
    #[test]
    fn compute_over_existing_lower_parts_is_not_gated() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        // Stands in for an array materialized from a file: the parts already exist.
        let array = deserialize_with(1, vec![msp(), lower_part()])
            .and_then(Array::try_from_parts)?
            .into_array();

        let sliced = array.slice(0..2)?;
        assert_eq!(sliced.execute::<DecimalArray>(&mut ctx)?.len(), 2);
        Ok(())
    }

    /// A crafted file may declare more lower parts than its precision needs. Assembling those
    /// parts would produce values outside the declared precision, so it must be rejected at
    /// deserialization rather than panicking later in the scalar path.
    #[test]
    fn test_deserialize_rejects_parts_wider_than_precision() {
        let result = deserialize_with(2, vec![msp(), lower_part(), lower_part()])
            .and_then(Array::try_from_parts);
        assert!(result.is_err(), "expected rejection, got {result:?}");
    }

    #[test]
    fn test_deserialize_rejects_child_count_mismatch() {
        // Metadata claiming a lower part that was not serialized.
        assert!(deserialize_with(1, vec![msp()]).is_err());
        // Metadata claiming fewer lower parts than there are children.
        assert!(deserialize_with(0, vec![msp(), lower_part()]).is_err());
        // Metadata claiming more lower parts than the encoding supports.
        assert!(
            deserialize_with(
                4,
                vec![
                    msp(),
                    lower_part(),
                    lower_part(),
                    lower_part(),
                    lower_part()
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn test_wide_decimal_buffer_types() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let i128_array = i128_parts(vec![1i128 << 100], Validity::NonNullable);
        let canonical = i128_array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(canonical.values_type(), DecimalType::I128);

        let i256_array = i256_parts(vec![i256_of(1 << 100, 0)], Validity::NonNullable);
        let canonical = i256_array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(canonical.values_type(), DecimalType::I256);

        // A narrow MSP with a single lower part still fits 128 bits.
        let array = DecimalByteParts::try_new_with_lower_parts(
            buffer![1i8, -1, 0].into_array(),
            vec![buffer![7u64, 7, 7].into_array()],
            DecimalDType::new(38, 2),
        )?;
        let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(canonical.values_type(), DecimalType::I128);
        assert_eq!(
            canonical.buffer::<i128>().as_slice(),
            &[(1i128 << 64) | 7, (-1i128 << 64) | 7, 7]
        );

        // Two lower parts under a narrow MSP overflow 128 bits, so the value widens.
        let array = DecimalByteParts::try_new_with_lower_parts(
            buffer![1i8].into_array(),
            vec![buffer![0u64].into_array(), buffer![9u64].into_array()],
            DecimalDType::new(76, 2),
        )?;
        let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(canonical.values_type(), DecimalType::I256);
        assert_eq!(canonical.buffer::<i256>().as_slice(), &[i256_of(1, 9)]);
        Ok(())
    }

    #[test]
    fn test_unused_buffer_of_values_is_ignored_for_null_rows() -> VortexResult<()> {
        // Null rows may hold arbitrary bits in the lower parts; they must stay null.
        let array = DecimalByteParts::try_new_with_lower_parts(
            PrimitiveArray::new(
                buffer![0i64, 0, 0],
                Validity::Array(BoolArray::from_iter([false, false, true]).into_array()),
            )
            .into_array(),
            vec![buffer![7u64, 9, 11].into_array()],
            DecimalDType::new(38, 2),
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            array.execute_scalar(0, &mut ctx)?,
            Scalar::null(array.dtype().clone())
        );
        let canonical = array.clone().execute::<DecimalArray>(&mut ctx)?;
        assert_arrays_eq!(array, canonical.into_array(), &mut ctx);
        Ok(())
    }
}
