// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod byte_length;
mod cast;
mod compare;
mod filter;
mod like;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::take_referenced_canonical;
use vortex_array::arrays::varbin::take_varbin;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;
use crate::FSSTArraySlotsExt;

impl TakeExecute for FSST {
    /// Gathers compressed rows for a sparse take, decodes each referenced row once for a
    /// repetitive one, and otherwise leaves execution to decode every row and gather the decoded
    /// strings.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_referenced_canonical(array.array(), indices, ctx, |ctx| {
            FSST::try_new_with_symbol_table(
                array
                    .dtype()
                    .clone()
                    .union_nullability(indices.dtype().nullability()),
                array.symbol_table(),
                take_varbin(array.codes().as_view(), indices, ctx)?,
                array
                    .uncompressed_lengths()
                    .take(indices.clone())?
                    .fill_null(Scalar::zero_value(
                        &array.uncompressed_lengths_dtype().clone(),
                    ))?,
                ctx,
            )
            .map(IntoArray::into_array)
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_take_null() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr =
            VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable)).into_array();
        let compr = fsst_train_compressor(&arr, &mut ctx)?;
        let fsst = fsst_compress(&arr, &compr, &mut ctx)?;

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            fsst.take(idx1.into_array())?.dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            fsst.take(idx2.into_array())?.dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
        Ok(())
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
    fn test_take_fsst_conformance(#[case] varbin: VarBinArray) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let varbin = varbin.into_array();
        let compressor = fsst_train_compressor(&varbin, &mut ctx)?;
        let array = fsst_compress(&varbin, &compressor, &mut ctx)?;
        test_take_conformance(&array.into_array(), &mut ctx);
        Ok(())
    }

    /// A dictionary over FSST values returns the right rows whichever way the take kernel goes:
    /// gathering compressed rows, decoding the referenced rows, or decoding every row. Null
    /// indices hold out-of-bounds values, which the kernel must ignore.
    #[rstest]
    #[case::sparse(500, 2_000)]
    #[case::referenced(1_500, 40)]
    #[case::every_row(2_000, 2_000)]
    #[case::many_per_row(8_000, 2_000)]
    fn test_dict_over_fsst_take(#[case] len: u32, #[case] range: u32) -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let input = VarBinArray::from_iter(
            (0..2_000)
                .map(|i| (i % 11 != 0).then(|| format!("https://www.example.com/items/{i:06}"))),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&input, &mut ctx)?;
        let fsst = fsst_compress(&input, &compressor, &mut ctx)?;

        let valid = (0..len).map(|i| i % 3 != 0).collect::<Vec<_>>();
        let values = (0..len)
            .zip(&valid)
            .map(|(i, &valid)| if valid { i * 7 % range } else { u32::MAX })
            .collect::<Buffer<u32>>();
        let indices = PrimitiveArray::new(values, Validity::from_iter(valid.iter().copied()));
        let dict = DictArray::try_new(indices.into_array(), fsst.into_array())?.into_array();

        let in_bounds = (0..len)
            .zip(&valid)
            .map(|(i, &valid)| valid.then_some(i * 7 % range));
        let expected = input.take(PrimitiveArray::from_option_iter(in_bounds).into_array())?;
        assert_arrays_eq!(dict, expected, &mut ctx);
        Ok(())
    }

    type FsstBuilder = fn(&mut ExecutionCtx) -> FSSTArray;

    #[rstest]
    // Basic string arrays
    #[case::fsst_simple(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Nullable strings
    #[case::fsst_nullable(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("test"), None],
            DType::Utf8(Nullability::Nullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Repetitive patterns (good for FSST compression)
    #[case::fsst_repetitive(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["http://example.com", "http://test.com", "http://vortex.dev", "http://data.org"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Edge cases
    #[case::fsst_single(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["single element"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    #[case::fsst_empty_strings(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["", "test", "", "hello", ""].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
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
        let array = VarBinArray::from_iter(data, DType::Utf8(Nullability::NonNullable)).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]

    fn test_fsst_consistency(#[case] build: FsstBuilder) {
        let mut ctx = array_session().create_execution_ctx();
        let array = build(&mut ctx);
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
