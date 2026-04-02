// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for VarBin {
    fn cast(array: ArrayView<'_, VarBin>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array
            .validity()
            .cast_nullability(new_nullability, array.len())?;
        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(Some(
            VarBinArray::try_new(
                array.offsets().clone(),
                array.bytes().clone(),
                new_dtype,
                new_validity,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::VarBinArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    #[case(
        DType::Utf8(Nullability::Nullable),
        DType::Utf8(Nullability::NonNullable)
    )]
    #[case(
        DType::Binary(Nullability::Nullable),
        DType::Binary(Nullability::NonNullable)
    )]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable)
    )]
    #[case(
        DType::Binary(Nullability::NonNullable),
        DType::Binary(Nullability::Nullable)
    )]
    fn try_cast_varbin_nullable(#[case] source: DType, #[case] target: DType) {
        let varbin = VarBinArray::from_iter(vec![Some("a"), Some("b"), Some("c")], source);

        let res = varbin.into_array().cast(target.clone());
        assert_eq!(res.unwrap().dtype(), &target);
    }

    #[rstest]
    #[should_panic]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[should_panic]
    #[case(DType::Binary(Nullability::Nullable))]
    fn try_cast_varbin_fail(#[case] source: DType) {
        let non_nullable_source = source.as_nonnullable();
        let varbin = VarBinArray::from_iter(vec![Some("a"), Some("b"), None], source);
        varbin.into_array().cast(non_nullable_source).unwrap();
    }

    #[rstest]
    #[case(VarBinArray::from_iter(vec![Some("hello"), Some("world"), Some("test")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinArray::from_iter(vec![Some("hello"), None, Some("world")], DType::Utf8(Nullability::Nullable)))]
    #[case(VarBinArray::from_iter(vec![Some(b"binary".as_slice()), Some(b"data".as_slice())], DType::Binary(Nullability::NonNullable)))]
    #[case(VarBinArray::from_iter(vec![Some(b"test".as_slice()), None], DType::Binary(Nullability::Nullable)))]
    #[case(VarBinArray::from_iter(vec![Some("single")], DType::Utf8(Nullability::NonNullable)))]
    fn test_cast_varbin_conformance(#[case] array: VarBinArray) {
        test_cast_conformance(&array.into_array());
    }
}
