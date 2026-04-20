// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;

use crate::Zstd;
use crate::ZstdData;
impl CastReduce for Zstd {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.eq_ignore_nullability(array.dtype()) {
            // Type changes can't be handled in ZSTD, need to decode and tweak.
            // TODO(aduffy): handle trivial conversions like Binary -> UTF8, integer widening, etc.
            return Ok(None);
        }

        let src_nullability = array.dtype().nullability();
        let target_nullability = dtype.nullability();

        match (src_nullability, target_nullability) {
            // Same type case. This should be handled in the layer above but for
            // completeness of the match arms we also handle it here.
            (Nullability::Nullable, Nullability::Nullable)
            | (Nullability::NonNullable, Nullability::NonNullable) => {
                Ok(Some(array.array().clone()))
            }
            (Nullability::NonNullable, Nullability::Nullable) => {
                // nonnull => null, trivial cast by altering the validity
                let unsliced_validity =
                    child_to_validity(&array.slots()[0], array.dtype().nullability());
                Ok(Some(
                    Zstd::try_new(
                        dtype.clone(),
                        ZstdData::new(
                            array.dictionary.clone(),
                            array.frames.clone(),
                            array.metadata.clone(),
                            array.unsliced_n_rows(),
                        ),
                        unsliced_validity,
                    )?
                    .into_array()
                    .slice(array.slice_start()..array.slice_stop())?,
                ))
            }
            (Nullability::Nullable, Nullability::NonNullable) => {
                // null => non-null works if there are no nulls in the sliced range
                let unsliced_validity =
                    child_to_validity(&array.slots()[0], array.dtype().nullability());
                let has_nulls = !matches!(
                    unsliced_validity.slice(array.slice_start()..array.slice_stop())?,
                    Validity::AllValid | Validity::NonNullable
                );

                // We don't attempt to handle casting when there are nulls.
                if has_nulls {
                    return Ok(None);
                }

                // If there are no nulls, the cast is trivial
                Ok(Some(
                    Zstd::try_new(
                        dtype.clone(),
                        ZstdData::new(
                            array.dictionary.clone(),
                            array.frames.clone(),
                            array.metadata.clone(),
                            array.unsliced_n_rows(),
                        ),
                        unsliced_validity,
                    )?
                    .into_array()
                    .slice(array.slice_start()..array.slice_stop())?,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::Zstd;

    #[test]
    fn test_cast_zstd_i32_to_i64() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let zstd = Zstd::from_primitive(&values, 0, 0, &mut ctx).unwrap();

        let casted = zstd
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([1i64, 2, 3, 4, 5]));
    }

    #[test]
    fn test_cast_zstd_nullability_change() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]);
        let zstd = Zstd::from_primitive(&values, 0, 0, &mut ctx).unwrap();

        let casted = zstd
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_sliced_zstd_nullable_to_nonnullable() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![10u32, 20, 30, 40, 50, 60],
            Validity::from_iter([true, true, true, true, true, true]),
        );
        let zstd = Zstd::from_primitive(&values, 0, 128, &mut ctx).unwrap();
        let sliced = zstd.slice(1..5).unwrap();
        let casted = sliced
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        // Verify the values are correct
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([20u32, 30, 40, 50]));
    }

    #[test]
    fn test_cast_sliced_zstd_part_valid_to_nonnullable() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter([
            None,
            Some(20u32),
            Some(30),
            Some(40),
            Some(50),
            Some(60),
        ]);
        let zstd = Zstd::from_primitive(&values, 0, 128, &mut ctx).unwrap();
        let sliced = zstd.slice(1..5).unwrap();
        let casted = sliced
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        let expected = PrimitiveArray::from_iter([20u32, 30, 40, 50]);
        assert_arrays_eq!(decoded, expected);
    }

    #[rstest]
    #[case::i32(PrimitiveArray::new(
        buffer![100i32, 200, 300, 400, 500],
        Validity::NonNullable,
    ))]
    #[case::f64(PrimitiveArray::new(
        buffer![1.1f64, 2.2, 3.3, 4.4, 5.5],
        Validity::NonNullable,
    ))]
    #[case::single(PrimitiveArray::new(
        buffer![42i64],
        Validity::NonNullable,
    ))]
    #[case::large(PrimitiveArray::new(
        buffer![0u32..1000],
        Validity::NonNullable,
    ))]
    fn test_cast_zstd_conformance(#[case] values: PrimitiveArray) {
        let zstd = Zstd::from_primitive(&values, 0, 0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        test_cast_conformance(&zstd.into_array());
    }
}
