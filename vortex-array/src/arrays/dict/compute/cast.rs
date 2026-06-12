// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Dict {
    fn cast(array: ArrayView<'_, Dict>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Can have un-reference null values making the cast of values fail without a possible mask.
        // TODO(joe): optimize this, could look at accessible values and fill_null not those?
        if !dtype.is_nullable() && !array.values().validity()?.definitely_no_nulls() {
            return Ok(None);
        }
        // Cast the dictionary values to the target type
        let casted_values = array.values().cast(dtype.clone())?;

        // If the codes are nullable but we are casting to non nullable dtype we have to remove nullability from codes as well
        let casted_codes = if array.codes().dtype().is_nullable() && !dtype.is_nullable() {
            array
                .codes()
                .cast(array.codes().dtype().with_nullability(dtype.nullability()))?
        } else {
            array.codes().clone()
        };

        // SAFETY: casting does not alter invariants of the codes
        Ok(Some(
            unsafe {
                DictArray::new_unchecked(casted_codes, casted_values)
                    .set_all_values_referenced(array.has_all_values_referenced())
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::Dict;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_cast_dict_to_wider_type() {
        let values = buffer![1i32, 2, 3, 2, 1].into_array();
        let dict = dict_encode(&values, &mut SESSION.create_execution_ctx()).unwrap();

        let casted = dict
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        #[expect(deprecated)]
        let decoded = casted.to_primitive();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([1i64, 2, 3, 2, 1]));
    }

    #[test]
    fn test_cast_dict_nullable() {
        let values =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(10), None]);
        let dict = dict_encode(&values.into_array(), &mut SESSION.create_execution_ctx()).unwrap();

        let casted = dict
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
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
        let dict = dict_encode(&values, &mut SESSION.create_execution_ctx()).unwrap();

        // Verify initial state - codes should be NonNullable, values should be NonNullable
        assert_eq!(dict.codes().dtype().nullability(), Nullability::NonNullable);
        assert_eq!(
            dict.values().dtype().nullability(),
            Nullability::NonNullable
        );

        // Cast to NonNullable (should be identity since already NonNullable)
        let non_nullable = dict
            .clone()
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            non_nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Check that codes and values are still NonNullable
        let non_nullable_dict = non_nullable.as_::<Dict>();
        assert_eq!(
            non_nullable_dict.codes().dtype().nullability(),
            Nullability::NonNullable
        );
        assert_eq!(
            non_nullable_dict.values().dtype().nullability(),
            Nullability::NonNullable
        );

        // Cast to Nullable
        let nullable = non_nullable
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Check that both codes and values are now Nullable
        let nullable_dict = nullable.as_::<Dict>();
        assert_eq!(
            nullable_dict.codes().dtype().nullability(),
            Nullability::NonNullable
        );
        assert_eq!(
            nullable_dict.values().dtype().nullability(),
            Nullability::Nullable
        );

        // Cast back to NonNullable
        let back_to_non_nullable = nullable
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            back_to_non_nullable.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Verify values are unchanged
        #[expect(deprecated)]
        let original_values = dict.as_array().to_primitive();
        #[expect(deprecated)]
        let final_values = back_to_non_nullable.to_primitive();
        assert_arrays_eq!(original_values, final_values);
    }

    #[rstest]
    #[case(dict_encode(&buffer![1i32, 2, 3, 2, 1, 3].into_array(), &mut SESSION.create_execution_ctx()).unwrap().into_array())]
    #[case(dict_encode(&buffer![100u32, 200, 100, 300, 200].into_array(), &mut SESSION.create_execution_ctx()).unwrap().into_array())]
    #[case(dict_encode(&PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).into_array(), &mut SESSION.create_execution_ctx()).unwrap().into_array())]
    #[case(dict_encode(&buffer![1.5f32, 2.5, 1.5, 3.5].into_array(), &mut SESSION.create_execution_ctx()).unwrap().into_array())]
    fn test_cast_dict_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(&array);
    }

    #[test]
    fn test_cast_dict_with_unreferenced_null_values_to_nonnullable() {
        use crate::arrays::DictArray;
        use crate::validity::Validity;

        // Create a dict with nullable values that have unreferenced null entries.
        // Values: [1.0, null, 3.0] (index 1 is null but no code points to it)
        // Codes: [0, 2, 0] (only reference indices 0 and 2, never 1)
        let values = PrimitiveArray::new(
            buffer![1.0f64, 0.0f64, 3.0f64],
            Validity::from(vortex_buffer::BitBuffer::from(vec![true, false, true])),
        )
        .into_array();
        let codes = buffer![0u32, 2, 0].into_array();
        let dict = DictArray::try_new(codes, values).unwrap();

        // The dict is Nullable (because values are nullable), but all codes point to valid values.
        assert_eq!(
            dict.dtype(),
            &DType::Primitive(PType::F64, Nullability::Nullable)
        );

        // Casting to NonNullable should succeed since all logical values are non-null.
        let result = dict
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::NonNullable));
        assert!(
            result.is_ok(),
            "cast to NonNullable should succeed for dict with only unreferenced null values"
        );
        let casted = result.unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );
        #[expect(deprecated)]
        let casted_prim = casted.to_primitive();
        assert_arrays_eq!(casted_prim, PrimitiveArray::from_iter([1.0f64, 3.0, 1.0]));
    }
}
