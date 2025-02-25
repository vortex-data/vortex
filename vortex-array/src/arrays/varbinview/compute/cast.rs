use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::CastFn;
use crate::{Array, ArrayRef};

impl CastFn<&VarBinViewArray> for VarBinViewEncoding {
    fn cast(&self, array: &VarBinViewArray, dtype: &DType) -> VortexResult<ArrayRef> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("Cannot cast {} to {}", array.dtype(), dtype);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array.validity().clone().cast_nullability(new_nullability)?;
        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(VarBinViewArray::try_new(
            array.views().clone(),
            array.buffers().to_vec(),
            new_dtype,
            new_validity,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::VarBinViewArray;
    use crate::compute::try_cast;

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
        let varbin = VarBinViewArray::from_iter(vec![Some("a"), Some("b"), Some("c")], source);

        let res = try_cast(&varbin, &target);
        assert_eq!(res.unwrap().dtype(), &target);
    }

    #[rstest]
    #[should_panic]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[should_panic]
    #[case(DType::Binary(Nullability::Nullable))]
    fn try_cast_varbin_fail(#[case] source: DType) {
        let non_nullable_source = source.as_nonnullable();
        let varbin = VarBinViewArray::from_iter(vec![Some("a"), Some("b"), None], source);
        try_cast(&varbin, &non_nullable_source).unwrap();
    }
}
