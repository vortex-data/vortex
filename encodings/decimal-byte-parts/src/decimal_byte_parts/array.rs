// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal byte-parts array types, validation, and VTable implementations.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

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
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
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
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use super::MAX_LOWER_PARTS;
use super::assemble::assemble_decimal;
use super::assemble::assemble_wide_decimal_value;
use super::decimal_byte_parts_v2_id;
use super::rules::PARENT_RULES;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

/// This array encodes decimals by splitting them between 1-4 columns of primitive typed children.
///
/// The most significant part (MSP) stores the most significant decimal bits. It is signed and is
/// nullable iff the decimal is nullable.
///
/// Every lower part is a non-nullable unsigned integer holding a 64-bit window of the value.
/// Parts may have narrower integer dtypes when their values fit; their positions stay 64 bits apart.
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

impl ArrayHash for DecimalBytePartsData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for DecimalBytePartsData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl DecimalBytePartsData {
    /// Validate the parts of a [`DecimalBytePartsArray`].
    ///
    /// # Errors
    ///
    /// Returns an error if the MSP is not a signed integer array of length `len`, if `dtype`
    /// does not match the MSP's nullability, if there are more than `MAX_LOWER_PARTS`
    /// lower parts, or if any lower part is not a non-nullable unsigned integer array of length `len`.
    pub fn validate<'a>(
        msp: &ArrayRef,
        lower_parts: impl ExactSizeIterator<Item = &'a ArrayRef>,
        decimal_dtype: DecimalDType,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        if !msp.dtype().is_signed_int() {
            vortex_bail!("msp must be a signed integer array")
        }

        let expected_dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        vortex_ensure!(
            dtype == &expected_dtype,
            "expected dtype {expected_dtype}, got {dtype}"
        );
        vortex_ensure!(msp.len() == len, "expected len {len}, got {}", msp.len());

        let lower_part_count = lower_parts.len();

        vortex_ensure!(
            lower_part_count <= MAX_LOWER_PARTS,
            "at most {MAX_LOWER_PARTS} lower parts are supported, got {lower_part_count}"
        );
        for (idx, part) in lower_parts.enumerate() {
            vortex_ensure!(
                part.dtype().is_unsigned_int() && !part.dtype().is_nullable(),
                "lower part {idx} must have a non-nullable unsigned integer dtype, got {}",
                part.dtype()
            );
            vortex_ensure!(
                part.len() == len,
                "lower part {idx} has len {}, expected {len}",
                part.len()
            );
        }
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
    /// Lower parts are ordered most significant first and must each be a non-nullable unsigned integer
    /// array of the same length as the MSP. See [`super::split_decimal`] for producing them from a
    /// decimal array.
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
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        let slots = DecimalBytePartsSlots { msp, lower_parts }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData).with_slots(slots),
        )
    }

    /// Construct a [`DecimalBytePartsArray`] from parts whose invariants are already established.
    ///
    /// # Safety
    ///
    /// The MSP must have a signed integer dtype (`i8`, `i16`, `i32`, or `i64`). There must be
    /// at most `MAX_LOWER_PARTS` lower parts, each a non-nullable unsigned integer array with the same
    /// length as the MSP. Lower parts are ordered most significant first.
    pub(super) unsafe fn new_unchecked(
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) -> DecimalBytePartsArray {
        let len = msp.len();
        let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
        let slots = DecimalBytePartsSlots { msp, lower_parts }.into_slots();
        // SAFETY: the caller guarantees the part types, lengths, and count. The slot builder
        // fills every required slot, and the length and nullability come from the MSP.
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(DecimalByteParts, dtype, len, DecimalBytePartsData)
                    .with_slots(slots),
            )
        }
    }
}

impl VTable for DecimalByteParts {
    type TypedArrayData = DecimalBytePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        decimal_byte_parts_v2_id()
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

        let min_slots = DecimalBytePartsSlots::FIXED_COUNT;
        let max_slots = min_slots + MAX_LOWER_PARTS;
        vortex_ensure!(
            (min_slots..=max_slots).contains(&slots.len()),
            "expected {min_slots}..={max_slots} slots, got {}",
            slots.len()
        );
        for (idx, slot) in slots.iter().enumerate() {
            vortex_ensure!(slot.is_some(), "missing required slot {idx}");
        }

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
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("DecimalByteParts serialization requires DecimalBytePartsPlugin")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("DecimalByteParts deserialization requires DecimalBytePartsPlugin")
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
        let lower_parts = array.lower_parts().to_vec();
        let assembled = assemble_decimal(array.msp(), &lower_parts, array.decimal_dtype(), ctx)?;

        Ok(ExecutionResult::done(assembled))
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

pub(crate) trait DecimalBytePartsArrayExt: DecimalBytePartsArraySlotsExt {
    /// The decimal dtype of this array.
    fn decimal_dtype(&self) -> DecimalDType {
        *self
            .as_ref()
            .dtype()
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype")
    }

    /// Rebuild the array by applying `f` to the MSP and every lower part, in slot order.
    ///
    /// This applies row operations such as slicing and filtering to all parts together,
    /// preserving the decimal precision and scale.
    fn map_parts(
        &self,
        mut f: impl FnMut(&ArrayRef) -> VortexResult<ArrayRef>,
    ) -> VortexResult<DecimalBytePartsArray> {
        let msp = f(self.msp())?;
        let lower_parts = self
            .lower_parts()
            .iter()
            .map(&mut f)
            .collect::<VortexResult<Vec<_>>>()?;
        DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, self.decimal_dtype())
    }

    /// Rebuild the array with a replacement MSP, preserving its lower parts, precision and scale.
    ///
    /// Use this for operations such as masking and nullability casts that only affect the MSP.
    /// The replacement MSP determines the result's nullability.
    fn with_msp(&self, msp: ArrayRef) -> VortexResult<DecimalBytePartsArray> {
        DecimalByteParts::try_new_with_lower_parts(
            msp,
            self.lower_parts().to_vec(),
            self.decimal_dtype(),
        )
    }
}

impl<T: TypedArrayRef<DecimalByteParts>> DecimalBytePartsArrayExt for T {}

impl OperationsVTable<DecimalByteParts> for DecimalByteParts {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, DecimalByteParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let scalar = array.msp().execute_scalar(index, ctx)?;

        // Widen the MSP's signed value (i8/i16/i32/i64) to i64 for scalar reconstruction.
        // The array retains its original MSP storage type.
        let primitive_scalar = scalar.as_primitive();
        let msp = primitive_scalar.as_::<i64>().vortex_expect("non-null");

        // Zero-extend each narrowed lower value to its 64-bit window.
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

        let value = match lower_parts.as_slice() {
            [] => DecimalValue::I64(msp),
            [first] => DecimalValue::I128(assemble_wide_decimal_value(msp, [*first])),
            [first, second] => {
                DecimalValue::I256(assemble_wide_decimal_value(msp, [*first, *second]))
            }
            [first, second, third] => {
                DecimalValue::I256(assemble_wide_decimal_value(msp, [*first, *second, *third]))
            }
            _ => vortex_bail!(
                "at most {MAX_LOWER_PARTS} lower parts are supported, got {}",
                lower_parts.len()
            ),
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
    use vortex_array::Array;
    use vortex_array::ArrayParts;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::DecimalType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::DecimalByteParts;
    use super::DecimalBytePartsArraySlotsExt;
    use super::DecimalBytePartsData;
    use crate::decimal_byte_parts::LOWER_PART_DTYPE;
    use crate::decimal_byte_parts::MAX_LOWER_PARTS;
    use crate::decimal_byte_parts::testing::i128_parts;
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
            vec![i256::from_parts((7u128 << 64) | 8, (5i128 << 64) | 6)],
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
    fn test_scalar_at_matches_canonical_for_each_part_count(
        #[values(false, true)] narrow_msp: bool,
        #[values(0, 1, 2, 3)] lower_count: usize,
    ) -> VortexResult<()> {
        let validity = Validity::from_iter([false, true, true]);
        let msp = if narrow_msp {
            PrimitiveArray::new(buffer![0i8, 3, -3], validity)
        } else {
            PrimitiveArray::new(buffer![0i64, 3, -3], validity)
        };
        let lower = [4u64, 1, 2]
            .into_iter()
            .take(lower_count)
            .map(|word| PrimitiveArray::new(buffer![word; 3], Validity::NonNullable).into_array())
            .collect();
        let dtype = DecimalDType::new(if lower_count <= 1 { 38 } else { 76 }, 0);
        let array = DecimalByteParts::try_new_with_lower_parts(msp.into_array(), lower, dtype)?;
        let mut ctx = array_session().create_execution_ctx();
        let canonical = array
            .clone()
            .into_array()
            .execute::<DecimalArray>(&mut ctx)?;
        for row in 0..array.len() {
            assert_eq!(
                array.execute_scalar(row, &mut ctx)?,
                canonical.execute_scalar(row, &mut ctx)?
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
    #[case([PType::U8, PType::U16, PType::U32])]
    #[case([PType::U16, PType::U32, PType::U64])]
    #[case([PType::U32, PType::U64, PType::U8])]
    #[case([PType::U64, PType::U8, PType::U16])]
    fn test_independently_narrowed_parts(
        #[case] lower_ptypes: [PType; 3],
        #[values(PType::I8, PType::I16, PType::I32, PType::I64)] msp_ptype: PType,
        #[values(1, 2, 3)] lower_count: usize,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let validity = Validity::from_iter([false, true, true]);
        let msp = PrimitiveArray::new(buffer![0i64, 1, -1], validity.clone())
            .into_array()
            .cast(DType::Primitive(msp_ptype, Nullability::Nullable))?;
        // Set the unsigned type's highest bit to catch accidental sign extension.
        let words = lower_ptypes.map(|ptype| 1u64 << (ptype.byte_width() * 8 - 1));
        let lower = lower_ptypes
            .into_iter()
            .zip(words)
            .take(lower_count)
            .map(|(ptype, word)| {
                buffer![0u64, word, word]
                    .into_array()
                    .cast(DType::Primitive(ptype, Nullability::NonNullable))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
        let encoded = DecimalByteParts::try_new_with_lower_parts(msp, lower, dtype)?;
        let expected = match lower_count {
            1 => DecimalArray::new(
                buffer![
                    0i128,
                    (1i128 << 64) | i128::from(words[0]),
                    (-1i128 << 64) | i128::from(words[0])
                ],
                dtype,
                validity,
            ),
            2 => {
                let low = (u128::from(words[0]) << 64) | u128::from(words[1]);
                DecimalArray::new(
                    buffer![
                        i256::ZERO,
                        i256::from_parts(low, 1),
                        i256::from_parts(low, -1)
                    ],
                    dtype,
                    validity,
                )
            }
            _ => {
                let low = (u128::from(words[1]) << 64) | u128::from(words[2]);
                DecimalArray::new(
                    buffer![
                        i256::ZERO,
                        i256::from_parts(low, (1i128 << 64) | i128::from(words[0])),
                        i256::from_parts(low, (-1i128 << 64) | i128::from(words[0])),
                    ],
                    dtype,
                    validity,
                )
            }
        };
        let actual = encoded
            .clone()
            .into_array()
            .execute::<DecimalArray>(&mut ctx)?;
        assert_arrays_eq!(expected.clone(), actual, &mut ctx);
        for row in 0..expected.len() {
            assert_eq!(
                encoded.execute_scalar(row, &mut ctx)?,
                expected.execute_scalar(row, &mut ctx)?
            );
        }
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
    fn test_rejects_invalid_parts(
        #[case] lower_parts: Vec<ArrayRef>,
        #[case] decimal_dtype: DecimalDType,
    ) {
        assert!(
            DecimalByteParts::try_new_with_lower_parts(msp(), lower_parts, decimal_dtype).is_err()
        );
    }

    #[rstest]
    #[case::no_slots(vec![])]
    #[case::missing_msp(vec![None])]
    #[case::missing_lower(vec![Some(msp()), None])]
    #[case::gap_in_lower(vec![Some(msp()), None, Some(lower_part())])]
    fn test_rejects_missing_slots(#[case] slots: Vec<Option<ArrayRef>>) {
        let parts = ArrayParts::new(
            DecimalByteParts,
            DType::Decimal(DecimalDType::new(76, 2), Nullability::NonNullable),
            3,
            DecimalBytePartsData,
        )
        .with_slots(slots.into_iter().collect());
        assert!(Array::try_from_parts(parts).is_err());
    }

    #[test]
    fn test_wide_decimal_buffer_types() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let i128_array = i128_parts(vec![1i128 << 100], Validity::NonNullable);
        let canonical = i128_array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(canonical.values_type(), DecimalType::I128);

        let i256_array = i256_parts(vec![i256::from_parts(0, 1 << 100)], Validity::NonNullable);
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
        assert_eq!(
            canonical.buffer::<i256>().as_slice(),
            &[i256::from_parts(9, 1)]
        );
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
