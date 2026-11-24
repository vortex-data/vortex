// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Casts from fixed-size binary arrays.
//!
//! A fixed-size binary array can change its nullability or become a variable-width binary array.
//! Casting to another byte width is not supported.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::scalar_fn::builtin::cast::CastKernel;
use crate::scalar_fn::builtin::cast::CastReduce;
use crate::validity::Validity;

fn with_validity(array: ArrayView<'_, FixedSizeBinary>, validity: Validity) -> ArrayRef {
    // SAFETY: Only the validity changes, and it keeps the length of the array.
    unsafe {
        FixedSizeBinaryArray::new_unchecked_handle(
            array.buffer_handle().clone(),
            array.byte_width(),
            array.len(),
            validity,
        )
    }
    .into_array()
}

impl CastReduce for FixedSizeBinary {
    fn cast(
        array: ArrayView<'_, FixedSizeBinary>,
        dtype: &DType,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let Some(validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        Ok(Some(with_validity(array, validity)))
    }
}

impl CastKernel for FixedSizeBinary {
    fn cast(
        array: ArrayView<'_, FixedSizeBinary>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match dtype {
            DType::FixedSizeBinary(byte_width, nullability)
                if *byte_width == array.byte_width() =>
            {
                let validity =
                    array
                        .validity()?
                        .cast_nullability(*nullability, array.len(), ctx)?;
                Ok(Some(with_validity(array, validity)))
            }
            DType::Binary(nullability) => {
                let validity =
                    array
                        .validity()?
                        .cast_nullability(*nullability, array.len(), ctx)?;
                let mask = validity.execute_mask(array.len(), ctx)?;
                let mut builder = VarBinViewBuilder::with_capacity_in(
                    dtype.clone(),
                    array.len(),
                    ctx.allocator().clone(),
                );
                for index in 0..array.len() {
                    if mask.value(index) {
                        builder.append_value(array.value(index));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Some(builder.finish()))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeBinaryArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn cast_nullability() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeBinaryArray::from_iter([b"ab", b"cd"], 2).into_array();
        let nullable = array.cast(DType::FixedSizeBinary(2, Nullability::Nullable))?;
        assert_eq!(
            nullable.dtype(),
            &DType::FixedSizeBinary(2, Nullability::Nullable)
        );
        let back = nullable.cast(DType::FixedSizeBinary(2, Nullability::NonNullable))?;
        assert_arrays_eq!(back, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn cast_nulls_to_non_nullable_fails() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), None], 2).into_array();
        let result = array
            .cast(DType::FixedSizeBinary(2, Nullability::NonNullable))
            .and_then(|array| array.execute::<FixedSizeBinaryArray>(&mut ctx));
        assert!(result.is_err());
    }

    #[test]
    fn cast_to_binary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeBinaryArray::from_option_iter([Some(b"ab"), None, Some(b"ef")], 2)
            .into_array()
            .cast(DType::Binary(Nullability::Nullable))?;
        let expected = VarBinViewArray::from_iter(
            [Some(b"ab".as_slice()), None, Some(b"ef".as_slice())],
            DType::Binary(Nullability::Nullable),
        );
        assert_arrays_eq!(array, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn cast_to_other_width_fails() {
        let mut ctx = SESSION.create_execution_ctx();
        let result = FixedSizeBinaryArray::from_iter([b"ab"], 2)
            .into_array()
            .cast(DType::FixedSizeBinary(3, Nullability::NonNullable))
            .and_then(|array| array.execute::<FixedSizeBinaryArray>(&mut ctx));
        assert!(result.is_err());
    }

    #[rstest]
    #[case::non_nullable(FixedSizeBinaryArray::from_iter([b"aa", b"bb", b"cc"], 2).into_array())]
    #[case::nullable(FixedSizeBinaryArray::from_option_iter(
        [Some(b"aaa"), None, Some(b"ccc")],
        3,
    ).into_array())]
    fn conformance(#[case] array: ArrayRef) {
        test_cast_conformance(&array, &mut SESSION.create_execution_ctx());
    }
}
