// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::dtype::ToBytes;
use vortex_array::match_each_float_ptype;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::BloomPartial;

pub(super) fn accumulate_primitive(
    array: &PrimitiveArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // TODO (joacoc): What about using density threshold?
    // TODO (joacoc): What about a single new macro that separates both:
    // match... {
    //      Floats => ..
    //      Integers => ..
    // }
    match array.ptype() {
        PType::F16 | PType::F32 | PType::F64 => accumulate_primitive_float(array, partial, ctx)?,
        _ => accumulate_primitive_int(array, partial, ctx)?,
    }

    Ok(())
}

fn accumulate_primitive_float(
    array: &PrimitiveArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_float_ptype!(array.ptype(), |T| {
        let slice = array.as_slice::<T>();

        match array.validity()?.execute_mask(slice.len(), ctx)? {
            Mask::AllTrue(_) => {
                for value in slice {
                    partial.insert(value.to_le_bytes());
                }
            }
            Mask::AllFalse(_) => {}
            Mask::Values(mask_values) => {
                for &(start, end) in mask_values.slices() {
                    for value in &slice[start..end] {
                        partial.insert(value.to_le_bytes());
                    }
                }
            }
        };
    });

    Ok(())
}

fn accumulate_primitive_int(
    array: &PrimitiveArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_integer_ptype!(array.ptype(), |T| {
        insert_integer_slice(array, partial, ctx, array.as_slice::<T>())?
    });

    Ok(())
}

fn insert_integer_slice<T>(
    array: &PrimitiveArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
    slice: &[T],
) -> VortexResult<()>
where
    T: ToBytes,
{
    match array.validity()?.execute_mask(slice.len(), ctx)? {
        Mask::AllTrue(_) => {
            for value in slice {
                partial.insert(value.to_le_bytes());
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask_values) => {
            for &(start, end) in mask_values.slices() {
                for value in &slice[start..end] {
                    partial.insert(value.to_le_bytes());
                }
            }
        }
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    #[cfg(test)]
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::build_filter;
    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::setup;

    #[rstest]
    #[case(&[10i8, 20, 30], 99i8)]
    #[case(&[10i16, 20, 30], 99i16)]
    #[case(&[10i32, 20, 30], 999i32)]
    #[case(&[10i64, 20, 30], 999i64)]
    #[case(&[10.0f32, 20.0, 30.0], 999.0f32)]
    #[case(&[10.0f64, 20.0, 30.0], 999.0f64)]
    fn membership<T>(#[case] present: &[T], #[case] absent: T) -> VortexResult<()>
    where
        T: Copy + NativePType + Into<PValue>,
    {
        let ctx = setup()?;
        let values = PrimitiveArray::from_iter(present.iter().copied());
        let bloom_filter = build_filter(
            values.into_array(),
            DType::Primitive(T::PTYPE, Nullability::NonNullable),
            ctx,
        )?;
        for &v in present {
            let scalar = Scalar::primitive(v, Nullability::NonNullable);
            assert!(bloom_filter.contains_scalar(&scalar)?);
        }
        let scalar = Scalar::primitive(absent, Nullability::NonNullable);
        assert!(!bloom_filter.contains_scalar(&scalar)?);
        Ok(())
    }

    /// The following three test will check if the validity cases
    /// are correctly implemented for the three branches.
    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_all_true<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + NativePType + Into<PValue>,
    {
        let ctx = setup()?;
        let all_valid = PrimitiveArray::from_option_iter(present.iter().map(|&v| Some(v)));
        let bloom_filter = build_filter(
            all_valid.into_array(),
            DType::Primitive(T::PTYPE, Nullability::Nullable),
            ctx,
        )?;

        for &v in present {
            let scalar = Scalar::primitive(v, Nullability::Nullable);
            assert!(bloom_filter.contains_scalar(&scalar)?);
        }

        Ok(())
    }

    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_all_false<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + NativePType + Into<PValue>,
    {
        let ctx = setup()?;
        let all_invalid = PrimitiveArray::from_option_iter(present.iter().map(|_| None::<T>));
        let bloom_filter = build_filter(
            all_invalid.into_array(),
            DType::Primitive(T::PTYPE, Nullability::Nullable),
            ctx,
        )?;

        for &v in present {
            let scalar = Scalar::primitive(v, Nullability::Nullable);
            assert!(!bloom_filter.contains_scalar(&scalar)?);
        }

        Ok(())
    }

    #[rstest]
    #[case(&[10i8, 20, 30, 40, 50])]
    fn validity_mixed<T>(#[case] present: &[T]) -> VortexResult<()>
    where
        T: Copy + NativePType + Into<PValue>,
    {
        let ctx = setup()?;
        let mixed: Vec<Option<T>> = present
            .iter()
            .enumerate()
            .map(|(i, &v)| (i % 2 == 0).then_some(v))
            .collect();

        let bloom_filter = build_filter(
            PrimitiveArray::from_option_iter(mixed).into_array(),
            DType::Primitive(T::PTYPE, Nullability::Nullable),
            ctx,
        )?;

        for (i, &v) in present.iter().enumerate() {
            if i % 2 == 0 {
                let scalar = Scalar::primitive(v, Nullability::Nullable);
                assert!(bloom_filter.contains_scalar(&scalar)?);
            }
        }

        Ok(())
    }

    // The idea is here is to test two different NaN float values.
    //
    // Given that the bloom filter uses bits, it means that one
    // NaN value could be present but others not.
    //
    // So the following test does the following, insert a NaN
    // value, update the NaN value to another NaN value with a different
    // set of bits, and then check that it doesn't exists.
    #[test]
    fn nan_bit_patterns_are_distinct_members() -> VortexResult<()> {
        let ctx = setup()?;
        let canonical_nan = f64::NAN;

        let present = [1.0_f64, canonical_nan, 3.0];
        let values = PrimitiveArray::from_iter(present);
        let bloom_filter = build_filter(
            values.into_array(),
            DType::Primitive(PType::F64, Nullability::NonNullable),
            ctx,
        )?;

        assert!(
            bloom_filter.contains_scalar(&Scalar::primitive(1.0_f64, Nullability::NonNullable))?
        );
        assert!(
            bloom_filter.contains_scalar(&Scalar::primitive(3.0_f64, Nullability::NonNullable))?
        );
        assert!(
            bloom_filter
                .contains_scalar(&Scalar::primitive(canonical_nan, Nullability::NonNullable))?
        );

        // Check that another NaN doesn't exists.
        // Update the canonical
        let other_nan = f64::from_bits(canonical_nan.to_bits() ^ 0x1);
        assert!(other_nan.is_nan());

        assert!(
            !bloom_filter
                .contains_scalar(&Scalar::primitive(other_nan, Nullability::NonNullable))?
        );
        assert!(
            !bloom_filter
                .contains_scalar(&Scalar::primitive(999.0_f64, Nullability::NonNullable))?
        );

        Ok(())
    }
}
