// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Sequence;
impl CastReduce for Sequence {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
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
                Sequence::try_new(
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
            let base_scalar = Scalar::try_new(
                DType::Primitive(array.ptype(), Nullability::NonNullable),
                Some(ScalarValue::Primitive(array.base())),
            )?;
            let multiplier_scalar = Scalar::try_new(
                DType::Primitive(array.ptype(), Nullability::NonNullable),
                Some(ScalarValue::Primitive(array.multiplier())),
            )?;

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
                Sequence::try_new(
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use crate::Sequence;
    use crate::SequenceArray;

    #[test]
    fn test_cast_sequence_nullability() {
        let sequence = Sequence::try_new_typed(0u32, 1u32, Nullability::NonNullable, 4).unwrap();

        // Cast to nullable
        let casted = sequence
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_sequence_u32_to_i64() {
        let sequence = Sequence::try_new_typed(100u32, 10u32, Nullability::NonNullable, 4).unwrap();

        let casted = sequence
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify the values
        #[expect(deprecated)]
        let decoded = casted.to_primitive();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([100i64, 110, 120, 130]));
    }

    #[test]
    fn test_cast_sequence_i16_to_i32_nullable() {
        // Test ptype change AND nullability change in one cast
        let sequence = Sequence::try_new_typed(5i16, 3i16, Nullability::NonNullable, 3).unwrap();

        let casted = sequence
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Verify the values
        #[expect(deprecated)]
        let decoded = casted.to_primitive();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_option_iter([Some(5i32), Some(8), Some(11)])
        );
    }

    #[test]
    fn test_cast_sequence_to_float_delegates_to_canonical() {
        let sequence = Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 5).unwrap();

        // Cast to float should delegate to canonical (SequenceArray doesn't support float)
        let casted = sequence
            .into_array()
            .cast(DType::Primitive(PType::F32, Nullability::NonNullable))
            .unwrap();
        // Should still succeed by decoding to canonical first
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F32, Nullability::NonNullable)
        );

        // Verify the values were correctly converted
        #[expect(deprecated)]
        let decoded = casted.to_primitive();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_iter([0.0f32, 1.0, 2.0, 3.0, 4.0])
        );
    }

    #[rstest]
    #[case::i32(Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 5).unwrap())]
    #[case::u64(Sequence::try_new_typed(1000u64, 100u64, Nullability::NonNullable, 4).unwrap())]
    // TODO(DK): SequenceArray does not actually conform. You cannot cast this array to u8 even
    // though all its values are representable therein.
    //
    // #[case::negative_step(Sequence::try_new_typed(100i32, -10i32, Nullability::NonNullable,
    // 5).unwrap())]
    #[case::single(Sequence::try_new_typed(42i64, 0i64, Nullability::NonNullable, 1).unwrap())]
    #[case::constant(Sequence::try_new_typed(
        100i32,
        0i32, // multiplier of 0 means constant array
        Nullability::NonNullable,
        5,
    ).unwrap())]
    fn test_cast_sequence_conformance(#[case] sequence: SequenceArray) {
        test_cast_conformance(&sequence.into_array());
    }
}
