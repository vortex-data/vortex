// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::FoRData;
use crate::r#for::FoR;
impl CastReduce for FoR {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // FoR only supports integer types
        if !dtype.is_int() {
            return Ok(None);
        }

        // For type changes between integers, cast the components
        let casted_child = array.encoded().cast(dtype.clone())?;
        let casted_reference = array.reference_scalar().cast(dtype)?;

        Ok(Some(
            FoRData::try_new(casted_child, casted_reference)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::FoRArray;
    use crate::FoRData;

    fn for_arr(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoRArray::try_from_data(FoRData::try_new(encoded, reference).unwrap())
            .vortex_expect("FoRData is always valid")
    }

    #[test]
    fn test_cast_for_i32_to_i64() {
        let for_array = for_arr(
            buffer![0i32, 10, 20, 30, 40].into_array(),
            Scalar::from(100i32),
        );

        let casted = for_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify the values after decoding
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([100i64, 110, 120, 130, 140])
        );
    }

    #[test]
    fn test_cast_for_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(20), Some(30), None]);
        let for_array = for_arr(values.into_array(), Scalar::from(50i32));

        let casted = for_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(for_arr(
        buffer![0i32, 1, 2, 3, 4].into_array(),
        Scalar::from(100i32)
    ))]
    #[case(for_arr(
        buffer![0u64, 10, 20, 30].into_array(),
        Scalar::from(1000u64)
    ))]
    #[case(for_arr(
        PrimitiveArray::from_option_iter([Some(0i16), None, Some(5), Some(10), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case(for_arr(
        buffer![-10i32, -5, 0, 5, 10].into_array(),
        Scalar::from(-100i32)
    ))]
    fn test_cast_for_conformance(#[case] array: FoRArray) {
        test_cast_conformance(&array.into_array());
    }
}
