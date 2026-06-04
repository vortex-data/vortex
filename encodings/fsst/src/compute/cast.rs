// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;

fn build_with_codes_validity(
    array: ArrayView<'_, FSST>,
    dtype: &DType,
    new_codes_validity: Validity,
) -> VortexResult<ArrayRef> {
    let codes = array.codes();
    let new_codes = VarBinArray::try_new(
        codes.offsets().clone(),
        codes.bytes().clone(),
        codes.dtype().with_nullability(dtype.nullability()),
        new_codes_validity,
    )?;

    Ok(unsafe {
        FSST::new_unchecked_with_symbol_table(
            dtype.clone(),
            array.symbol_table(),
            new_codes,
            array.uncompressed_lengths().clone(),
        )
    }
    .into_array())
}

impl CastReduce for FSST {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let codes = array.codes();
        let Some(new_codes_validity) = codes
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), codes.len())?
        else {
            return Ok(None);
        };

        Ok(Some(build_with_codes_validity(
            array,
            dtype,
            new_codes_validity,
        )?))
    }
}

impl CastKernel for FSST {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let codes = array.codes();
        let new_codes_validity =
            codes
                .validity()?
                .cast_nullability(dtype.nullability(), codes.len(), ctx)?;

        Ok(Some(build_with_codes_validity(
            array,
            dtype,
            new_codes_validity,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_session::VortexSession;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_cast_fsst_nullability() {
        let mut ctx = SESSION.create_execution_ctx();
        let strings = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), Some("hello world")],
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = fsst_train_compressor(&strings);
        let len = strings.len();
        let dtype = strings.dtype().clone();
        let fsst = fsst_compress(strings, len, &dtype, &compressor, &mut ctx);

        // Cast to nullable
        let casted = fsst
            .into_array()
            .cast(DType::Utf8(Nullability::Nullable))
            .unwrap();
        assert_eq!(casted.dtype(), &DType::Utf8(Nullability::Nullable));
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        vec![Some("hello"), Some("world"), Some("hello world")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("foo"), None, Some("bar"), Some("foobar")],
        DType::Utf8(Nullability::Nullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("test")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    fn test_cast_fsst_conformance(#[case] array: VarBinArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = fsst_train_compressor(&array);
        let fsst = fsst_compress(&array, array.len(), array.dtype(), &compressor, &mut ctx);
        test_cast_conformance(&fsst.into_array());
    }
}
