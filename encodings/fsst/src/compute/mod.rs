// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
mod like;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FSST;
use crate::FSSTArrayExt;

impl TakeExecute for FSST {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FSST::try_new(
                array
                    .dtype()
                    .clone()
                    .union_nullability(indices.dtype().nullability()),
                array.symbols().clone(),
                array.symbol_lengths().clone(),
                {
                    let codes = array.codes();
                    let codes = codes.as_view();
                    <VarBin as TakeExecute>::take(codes, indices, ctx)?
                        .vortex_expect("VarBin take kernel always returns Some")
                }
                .try_downcast::<VarBin>()
                .map_err(|_| vortex_err!("take for codes must return varbin array"))?,
                array
                    .uncompressed_lengths()
                    .take(indices.clone())?
                    .fill_null(Scalar::zero_value(
                        &array.uncompressed_lengths_dtype().clone(),
                    ))?,
                ctx,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_take_null() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));
        let compr = fsst_train_compressor(&arr);
        let fsst = fsst_compress(&arr, arr.len(), arr.dtype(), &compr, &mut ctx);

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            fsst.take(idx1.into_array()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            fsst.take(idx2.into_array()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("world"), Some("test"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        ["single element"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    fn test_take_fsst_conformance(#[case] varbin: VarBinArray) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let compressor = fsst_train_compressor(&varbin);
        let array = fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, &mut ctx);
        test_take_conformance(&array.into_array());
    }

    type FsstBuilder = fn(&mut ExecutionCtx) -> FSSTArray;

    #[rstest]
    // Basic string arrays
    #[case::fsst_simple(|ctx: &mut ExecutionCtx| {
        let varbin = VarBinArray::from_iter(
            ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
    })]
    // Nullable strings
    #[case::fsst_nullable(|ctx: &mut ExecutionCtx| {
        let varbin = VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("test"), None],
            DType::Utf8(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        fsst_compress(varbin, len, &dtype, &compressor, ctx)
    })]
    // Repetitive patterns (good for FSST compression)
    #[case::fsst_repetitive(|ctx: &mut ExecutionCtx| {
        let varbin = VarBinArray::from_iter(
            ["http://example.com", "http://test.com", "http://vortex.dev", "http://data.org"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
    })]
    // Edge cases
    #[case::fsst_single(|ctx: &mut ExecutionCtx| {
        let varbin = VarBinArray::from_iter(
            ["single element"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, varbin.len(), varbin.dtype(), &compressor, ctx)
    })]
    #[case::fsst_empty_strings(|ctx: &mut ExecutionCtx| {
        let varbin = VarBinArray::from_iter(
            ["", "test", "", "hello", ""].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        fsst_compress(varbin, len, &dtype, &compressor, ctx)
    })]
    // Large arrays
    #[case::fsst_large(|ctx: &mut ExecutionCtx| {
        let data: Vec<Option<&str>> = (0..1500)
            .map(|i| Some(match i % 10 {
                0 => "https://www.example.com/page",
                1 => "https://www.test.org/data",
                2 => "https://www.vortex.dev/docs",
                3 => "https://www.github.com/apache/arrow",
                4 => "https://www.rust-lang.org/learn",
                5 => "SELECT * FROM table WHERE id = ",
                6 => "INSERT INTO users (name, email) VALUES",
                7 => "UPDATE records SET status = 'active'",
                8 => "DELETE FROM logs WHERE timestamp < ",
                _ => "CREATE TABLE data (id INT, value TEXT)",
            }))
            .collect();
        let varbin = VarBinArray::from_iter(data, DType::Utf8(Nullability::NonNullable));
        let compressor = fsst_train_compressor(&varbin);
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        fsst_compress(varbin, len, &dtype, &compressor, ctx)
    })]

    fn test_fsst_consistency(#[case] build: FsstBuilder) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = build(&mut ctx);
        test_array_consistency(&array.into_array());
    }
}
