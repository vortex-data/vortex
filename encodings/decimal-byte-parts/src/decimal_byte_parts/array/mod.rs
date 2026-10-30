// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The DecimalByteParts encoding and its validated part layout.
//!
//! Validation bounds the part layout by the decimal precision, so decoding never narrows a value
//! assembled from lower parts. The most significant part (MSP) may remain narrow until a consumer
//! needs the canonical array.

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
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::require_child;
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
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use super::LOWER_PART_DTYPE;
use super::MAX_I128_LOWER_PARTS;
use super::MAX_I256_LOWER_PARTS;
use super::MAX_LOWER_PARTS;
use super::assemble::assemble_decimal;
use super::assemble::assemble_wide_decimal_value;
use super::decimal_byte_parts_v2_id;
use super::rules::PARENT_RULES;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

/// This array encodes decimals by splitting them between 1-4 columns of primitive typed children.
///
/// The most significant part (MSP) stores the most significant decimal bits. It is a signed integer
/// of at most 64 bits, and it is nullable iff the decimal is nullable. Only the MSP may use a
/// narrower dtype when its values fit.
///
/// Every lower part is a non-nullable `u64` array holding a 64-bit window of the value. The lower
/// parts never assemble wider than the canonical width of the decimal dtype: precision up to 18
/// allows no lower parts, precision 19 to 38 allows at most one, and precision 39 to 76 allows
/// up to three.
///
/// Executing the array decodes it into the canonical decimal array or a decimal [`Widen`], as
/// [`assemble_decimal`] describes.
///
/// e.g. for a decimal i128 \[ 127..64 | 63..0 \] msp = 127..64 and lower_part\[0\] = 63..0
///
/// All parts live in slots, so the array carries no additional data.
///
/// [`Widen`]: vortex_array::arrays::Widen
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
    /// does not match the MSP's nullability, if the lower parts assemble wider than the canonical
    /// width of `decimal_dtype`, or if any lower part is not a non-nullable `u64` array of length
    /// `len`.
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
        vortex_ensure_eq!(dtype, &expected_dtype);
        vortex_ensure_eq!(msp.len(), len);

        let lower_part_count = lower_parts.len();
        let max_lower_parts = max_lower_parts(decimal_dtype);
        vortex_ensure!(
            lower_part_count <= max_lower_parts,
            "{decimal_dtype} allows at most {max_lower_parts} lower parts, got {lower_part_count}"
        );

        for (idx, part) in lower_parts.enumerate() {
            vortex_ensure!(
                part.dtype() == &LOWER_PART_DTYPE,
                "lower part {idx} must have dtype {LOWER_PART_DTYPE}, got {}",
                part.dtype()
            );
            vortex_ensure_eq!(part.len(), len, "lower part {idx} length mismatch");
        }
        Ok(())
    }
}

/// The largest number of lower parts that assemble to at most the canonical width of
/// `decimal_dtype`.
///
/// One lower part assembles to `i128`, and two or three assemble to `i256`. An MSP is at most
/// `i64`, so a canonical width of `i64` or less allows no lower parts. With this bound, decoding
/// never narrows values that it assembled from lower parts.
fn max_lower_parts(decimal_dtype: DecimalDType) -> usize {
    match decimal_dtype.canonical_values_type() {
        DecimalType::I8 | DecimalType::I16 | DecimalType::I32 | DecimalType::I64 => 0,
        DecimalType::I128 => MAX_I128_LOWER_PARTS,
        DecimalType::I256 => MAX_I256_LOWER_PARTS,
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
    /// Lower parts are ordered most significant first and must each be a non-nullable `u64` array
    /// of the same length as the MSP. See [`super::split_decimal`] for producing them from a
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
        Array::try_from_parts(ArrayParts::new(
            DecimalByteParts,
            dtype,
            len,
            DecimalBytePartsData,
            slots,
        ))
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
        Ok(None)
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
        let mut array = require_child!(array, array.msp(), DecimalBytePartsSlots::MSP => Primitive);
        for idx in 0..array.lower_parts().len() {
            array = require_child!(
                array,
                array.lower_parts()[idx],
                DecimalBytePartsSlots::LOWER_PARTS_OFFSET + idx => Primitive
            );
        }
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
mod tests;
