// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{SequenceArray, SequenceVTable};

impl CastKernel for SequenceVTable {
    fn cast(&self, array: &SequenceArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // SequenceArray represents arithmetic sequences (base + i * multiplier) which
        // only makes sense for integer types. Floating-point sequences would accumulate
        // rounding errors, and other types don't support arithmetic operations.
        let DType::Primitive(target_ptype, target_nullability) = dtype else {
            return Ok(None);
        };

        if !target_ptype.is_int() {
            return Ok(None);
        }

        // Check if this is just a nullability change
        if array.ptype() == *target_ptype && array.dtype().nullability() != *target_nullability {
            // For SequenceArray, we can just create a new one with the same parameters
            // but different nullability
            return Ok(Some(
                SequenceArray::new(
                    array.base(),
                    array.multiplier(),
                    *target_ptype,
                    *target_nullability,
                    array.len(),
                )?
                .into_array(),
            ));
        }

        // For type changes, we need to cast the base and multiplier
        if array.ptype() != *target_ptype {
            // Create scalars from PValues and cast them
            let base_scalar = Scalar::new(
                DType::Primitive(array.ptype(), Nullability::NonNullable),
                ScalarValue::from(array.base()),
            );
            let multiplier_scalar = Scalar::new(
                DType::Primitive(array.ptype(), Nullability::NonNullable),
                ScalarValue::from(array.multiplier()),
            );

            let new_base_scalar =
                base_scalar.cast(&DType::Primitive(*target_ptype, Nullability::NonNullable))?;
            let new_multiplier_scalar = multiplier_scalar
                .cast(&DType::Primitive(*target_ptype, Nullability::NonNullable))?;

            // Extract PValues from the casted scalars
            let new_base = new_base_scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("Cast resulted in null base value"))?;
            let new_multiplier = new_multiplier_scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("Cast resulted in null multiplier value"))?;

            return Ok(Some(
                SequenceArray::new(
                    new_base,
                    new_multiplier,
                    *target_ptype,
                    *target_nullability,
                    array.len(),
                )?
                .into_array(),
            ));
        }

        Ok(None)
    }
}

register_kernel!(CastKernelAdapter(SequenceVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{ToCanonical, assert_arrays_eq};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::SequenceArray;

    #[test]
    fn test_cast_sequence_nullability() {
        let sequence = SequenceArray::typed_new(0u32, 1u32, Nullability::NonNullable, 4).unwrap();

        // Cast to nullable
        let casted = cast(
            sequence.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_sequence_u32_to_i64() {
        let sequence =
            SequenceArray::typed_new(100u32, 10u32, Nullability::NonNullable, 4).unwrap();

        let casted = cast(
            sequence.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify the values
        let decoded = casted.to_primitive();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([100i64, 110, 120, 130]));
    }

    #[test]
    fn test_cast_sequence_i16_to_i32_nullable() {
        // Test ptype change AND nullability change in one cast
        let sequence = SequenceArray::typed_new(5i16, 3i16, Nullability::NonNullable, 3).unwrap();

        let casted = cast(
            sequence.as_ref(),
            &DType::Primitive(PType::I32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Verify the values
        let decoded = casted.to_primitive();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_option_iter([Some(5i32), Some(8), Some(11)])
        );
    }

    #[test]
    fn test_cast_sequence_to_float_delegates_to_canonical() {
        let sequence = SequenceArray::typed_new(0i32, 1i32, Nullability::NonNullable, 5).unwrap();

        // Cast to float should delegate to canonical (SequenceArray doesn't support float)
        let casted = cast(
            sequence.as_ref(),
            &DType::Primitive(PType::F32, Nullability::NonNullable),
        )
        .unwrap();
        // Should still succeed by decoding to canonical first
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F32, Nullability::NonNullable)
        );

        // Verify the values were correctly converted
        let decoded = casted.to_primitive();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_iter([0.0f32, 1.0, 2.0, 3.0, 4.0])
        );
    }

    #[rstest]
    #[case::i32(SequenceArray::typed_new(0i32, 1i32, Nullability::NonNullable, 5).unwrap())]
    #[case::u64(SequenceArray::typed_new(1000u64, 100u64, Nullability::NonNullable, 4).unwrap())]
    #[case::negative_step(SequenceArray::typed_new(100i32, -10i32, Nullability::NonNullable, 5).unwrap())]
    #[case::single(SequenceArray::typed_new(42i64, 0i64, Nullability::NonNullable, 1).unwrap())]
    #[case::constant(SequenceArray::typed_new(
        100i32,
        0i32, // multiplier of 0 means constant array
        Nullability::NonNullable,
        5,
    ).unwrap())]
    fn test_cast_sequence_conformance(#[case] sequence: SequenceArray) {
        test_cast_conformance(sequence.as_ref());
    }
}
