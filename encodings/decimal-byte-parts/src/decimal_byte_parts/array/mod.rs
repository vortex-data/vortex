// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The DecimalByteParts encoding and its validated part layout.
//!
//! Construction establishes the [part layout](crate). The vtable and row operations preserve it.

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
use vortex_array::arrays::Widen;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::require_child;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
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

use super::LOWER_PART_DTYPE;
use super::MAX_LOWER_PARTS;
use super::assemble::assemble_decimal;
use super::assemble::assemble_wide_decimal_value;
use super::decimal_byte_parts_v2_id;
use super::max_lower_parts;
use super::msp_ptype;

/// A [`DecimalByteParts`]-encoded Vortex array.
pub type DecimalBytePartsArray = Array<DecimalByteParts>;

/// Decimal values stored in independently compressed integer parts.
///
/// See the [part layout](crate) and [`DecimalBytePartsData::validate`] for construction
/// requirements. All parts live in child slots. Valid values must fit the decimal precision.
/// Construction checks the layout, not the values.
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
    /// Returns an error unless the MSP has the logical ptype from the [part layout](crate), the
    /// nullability of `dtype`, and length `len`. The decimal dtype must match `decimal_dtype`.
    /// Every lower part must be a non-nullable `u64` array of length `len`, and their count must
    /// fit the canonical decimal width.
    pub fn validate<'a>(
        msp: &ArrayRef,
        lower_parts: impl ExactSizeIterator<Item = &'a ArrayRef>,
        decimal_dtype: DecimalDType,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        let expected_msp_dtype = DType::Primitive(msp_ptype(decimal_dtype), dtype.nullability());
        vortex_ensure_eq!(
            msp.dtype(),
            &expected_msp_dtype,
            "the MSP of {decimal_dtype} must have the integer dtype of its precision and the array \
             nullability; wrap a narrower integer MSP in a Widen"
        );
        vortex_ensure_eq!(dtype, &DType::Decimal(decimal_dtype, dtype.nullability()));
        vortex_ensure_eq!(msp.len(), len);

        let lower_part_count = lower_parts.len();
        let max_lower_parts = max_lower_parts(decimal_dtype);
        vortex_ensure!(
            lower_part_count <= max_lower_parts,
            "{decimal_dtype} allows at most {max_lower_parts} lower parts, got {lower_part_count}"
        );

        for (idx, part) in lower_parts.enumerate() {
            vortex_ensure_eq!(part.dtype(), &LOWER_PART_DTYPE, "lower part {idx} dtype");
            vortex_ensure_eq!(part.len(), len, "lower part {idx} length mismatch");
        }

        Ok(())
    }
}

/// The encoding vtable for decimal byte parts.
#[derive(Clone, Debug)]
pub struct DecimalByteParts;

impl DecimalByteParts {
    /// Construct a new [`DecimalBytePartsArray`] from an MSP array and decimal dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the MSP violates [`DecimalBytePartsData::validate`].
    pub fn try_new(
        msp: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<DecimalBytePartsArray> {
        Self::try_new_with_lower_parts(msp, Vec::new(), decimal_dtype)
    }

    /// Construct a new [`DecimalBytePartsArray`] from an MSP array, its lower parts, and a
    /// decimal dtype.
    ///
    /// Lower parts are ordered most significant first. See [`super::split_decimal`] for producing
    /// them from a decimal array.
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DecimalBytePartsSlots::slot_name(idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let mut array = if array.msp().is::<Widen>() {
            // Keep the MSP wrapped so `assemble_decimal` can read its narrow child. Executing the
            // `Widen` here would add an intermediate widening copy.
            array
        } else {
            require_child!(array, array.msp(), DecimalBytePartsSlots::MSP => Primitive)
        };

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

/// The integer children of a [`DecimalBytePartsArray`].
#[array_slots(DecimalByteParts)]
pub struct DecimalBytePartsSlots {
    /// The most significant parts of the decimal values.
    #[slot(0)]
    pub msp: ArrayRef,
    /// The remaining 64-bit windows of the decimal values, most significant first.
    #[slot(1..)]
    pub lower_parts: Vec<ArrayRef>,
}

/// Operations that preserve the validated decimal part layout.
pub(crate) trait DecimalBytePartsArrayExt: TypedArrayRef<DecimalByteParts> {
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

    /// Rebuild the array with a replacement MSP, preserving its lower parts, precision, and scale.
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

        Scalar::try_decimal(value, array.decimal_dtype(), array.dtype().nullability())
    }
}

impl ValidityChild<DecimalByteParts> for DecimalByteParts {
    fn validity_child(array: ArrayView<'_, DecimalByteParts>) -> ArrayRef {
        array.msp().clone()
    }
}

#[cfg(test)]
mod tests;
