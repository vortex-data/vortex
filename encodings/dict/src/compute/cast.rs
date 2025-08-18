// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{DictArray, DictVTable};

impl CastKernel for DictVTable {
    fn cast(&self, array: &DictArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast the dictionary values to the target type
        let casted_values = cast(array.values(), dtype)?;

        let casted_codes = if dtype.nullability() != array.codes().dtype().nullability() {
            cast(
                array.codes(),
                &array.codes().dtype().with_nullability(dtype.nullability()),
            )?
        } else {
            array.codes().clone()
        };

        // SAFETY: casting does not alter invariants of the codes
        unsafe {
            Ok(Some(
                DictArray::new_unchecked(casted_codes, casted_values).into_array(),
            ))
        }
    }
}

register_kernel!(CastKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::DictVTable;
    use crate::builders::dict_encode;

    #[test]
    fn test_cast_dict_to_wider_type() {
        let values = buffer![1i32, 2, 3, 2, 1].into_array();
        let dict = dict_encode(&values).unwrap();

        let casted = cast(
            dict.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i64>(), &[1i64, 2, 3, 2, 1]);
    }

    #[test]
    fn test_cast_dict_nullable() {
        let values =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(10), None]);
        let dict = dict_encode(values.as_ref()).unwrap();

        let casted = cast(
            dict.as_ref(),
            &DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_dict_allvalid_to_nonnullable_and_back() {
        // Create an AllValid dict array (no nulls)
        let values = buffer![10i32, 20, 30, 40].into_array();
        let dict = dict_encode(&values).unwrap();

        // Verify initial state - codes should be NonNullable, values should be NonNullable
        assert_eq!(dict.codes().dtype().nullability(), Nullability::NonNullable);
        assert_eq!(
            dict.values().dtype().nullability(),
            Nullability::NonNullable
        );

        // Cast to NonNullable (should be identity since already NonNullable)
        let non_nullable = cast(
            dict.as_ref(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            non_nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Check that codes and values are still NonNullable
        let non_nullable_dict = non_nullable.as_::<DictVTable>();
        assert_eq!(
            non_nullable_dict.codes().dtype().nullability(),
            Nullability::NonNullable
        );
        assert_eq!(
            non_nullable_dict.values().dtype().nullability(),
            Nullability::NonNullable
        );

        // Cast to Nullable
        let nullable = cast(
            non_nullable.as_ref(),
            &DType::Primitive(PType::I32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Check that both codes and values are now Nullable
        let nullable_dict = nullable.as_::<DictVTable>();
        assert_eq!(
            nullable_dict.codes().dtype().nullability(),
            Nullability::Nullable
        );
        assert_eq!(
            nullable_dict.values().dtype().nullability(),
            Nullability::Nullable
        );

        // Cast back to NonNullable
        let back_to_non_nullable = cast(
            nullable.as_ref(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            back_to_non_nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Check that both codes and values are NonNullable again
        let back_dict = back_to_non_nullable.as_::<DictVTable>();
        assert_eq!(
            back_dict.codes().dtype().nullability(),
            Nullability::NonNullable
        );
        assert_eq!(
            back_dict.values().dtype().nullability(),
            Nullability::NonNullable
        );

        // Verify values are unchanged
        let original_values = dict.to_canonical().unwrap().into_primitive().unwrap();
        let final_values = back_dict.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(
            original_values.as_slice::<i32>(),
            final_values.as_slice::<i32>()
        );
    }

    #[rstest]
    #[case(dict_encode(&buffer![1i32, 2, 3, 2, 1, 3].into_array()).unwrap().into_array())]
    #[case(dict_encode(&buffer![100u32, 200, 100, 300, 200].into_array()).unwrap().into_array())]
    #[case(dict_encode(&PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).into_array()).unwrap().into_array())]
    #[case(dict_encode(&buffer![1.5f32, 2.5, 1.5, 3.5].into_array()).unwrap().into_array())]
    fn test_cast_dict_conformance(#[case] array: vortex_array::ArrayRef) {
        test_cast_conformance(array.as_ref());
    }
}
