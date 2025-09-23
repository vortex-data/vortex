// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for VarBinViewVTable {
    fn cast(&self, array: &VarBinViewArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array
            .validity()
            .clone()
            .cast_nullability(new_nullability, array.len())?;
        let new_dtype = array.dtype().with_nullability(new_nullability);

        // SAFETY: casting just changes the DType, does not affect invariants on views/buffers.
        unsafe {
            Ok(Some(
                VarBinViewArray::new_unchecked(
                    array.views().clone(),
                    array.buffers().clone(),
                    new_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

register_kernel!(CastKernelAdapter(VarBinViewVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::VarBinViewArray;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;

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

        let res = cast(varbin.as_ref(), &target);
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
        cast(varbin.as_ref(), &non_nullable_source).unwrap();
    }

    #[rstest]
    #[case(VarBinViewArray::from_iter(vec![Some("hello"), Some("world"), Some("test")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("hello"), None, Some("world")], DType::Utf8(Nullability::Nullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some(b"binary".as_slice()), Some(b"data".as_slice())], DType::Binary(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some(b"test".as_slice()), None], DType::Binary(Nullability::Nullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("single")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("very long string that exceeds the inline size to test view functionality with multiple buffers")], DType::Utf8(Nullability::NonNullable)))]
    fn test_cast_varbinview_conformance(#[case] array: VarBinViewArray) {
        test_cast_conformance(array.as_ref());
    }
}
