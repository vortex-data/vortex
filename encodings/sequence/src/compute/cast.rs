// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

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

        if array.dtype() == dtype {
            return Ok(None);
        }

        // try_new also validates that the produced values fit the target ptype.
        Ok(Some(
            Sequence::try_new(
                array.base(),
                array.multiplier(),
                *target_ptype,
                *target_nullability,
                array.len(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sequence;
    use crate::SequenceArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

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
        let mut ctx = SESSION.create_execution_ctx();
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
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_iter([100i64, 110, 120, 130]),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_sequence_i16_to_i32_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_option_iter([Some(5i32), Some(8), Some(11)]),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_sequence_to_float_delegates_to_canonical() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_iter([0.0f32, 1.0, 2.0, 3.0, 4.0]),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_sequence_keeps_arithmetic_ptype_but_scalar_uses_output_dtype() -> VortexResult<()>
    {
        // Cast the public dtype to u8
        let casted = Sequence::try_new_typed(100i32, -10i32, Nullability::NonNullable, 5)?
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))?;

        let sequence = casted
            .as_typed::<Sequence>()
            .expect("integer sequence cast should preserve SequenceArray");
        assert_eq!(sequence.calculation_ptype(), PType::I64);
        assert_eq!(
            sequence.dtype(),
            &DType::Primitive(PType::U8, Nullability::NonNullable)
        );

        let scalar = casted.execute_scalar(1, &mut SESSION.create_execution_ctx())?;
        assert_eq!(
            scalar,
            Scalar::try_new(
                DType::Primitive(PType::U8, Nullability::NonNullable),
                Some(ScalarValue::from(90u8)),
            )?
        );

        Ok(())
    }

    #[rstest]
    #[case::i32(Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 5).unwrap())]
    #[case::u64(Sequence::try_new_typed(1000u64, 100u64, Nullability::NonNullable, 4).unwrap())]
    #[case::negative_step(Sequence::try_new_typed(100i32, -10i32, Nullability::NonNullable,
    5).unwrap())]
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
