use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{VarBinViewArray, VarBinViewEncoding};
use crate::compute::CastFn;
use crate::validity::Validity;
use crate::{Array, IntoArray};

impl CastFn<VarBinViewArray> for VarBinViewEncoding {
    fn cast(&self, array: &VarBinViewArray, dtype: &DType) -> VortexResult<Array> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("Cannot cast {} to {}", array.dtype(), dtype);
        }

        // If the types are the same, return the array,
        // otherwise set the array nullability as the dtype nullability.
        if dtype.is_nullable() || array.all_valid()? {
            VarBinViewArray::try_new(
                array.views(),
                array.buffers().collect(),
                dtype.clone(),
                if dtype.is_nullable() {
                    Validity::AllValid
                } else {
                    Validity::NonNullable
                },
            )
            .map(|a| a.into_array())
        } else {
            vortex_bail!("Cannot cast null array to non-nullable type");
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::VarBinViewArray;
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

        let res = try_cast(varbin, &target);
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
        try_cast(varbin, &non_nullable_source).unwrap();
    }
}
