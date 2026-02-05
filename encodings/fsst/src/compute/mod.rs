// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::TakeReduce;
use vortex_array::arrays::TakeReduceAdaptor;
use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::fill_null;
use vortex_array::compute::take;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;

use crate::FSSTArray;
use crate::FSSTVTable;

fn take_fsst(array: &FSSTArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
    Ok(FSSTArray::try_new(
        array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability()),
        array.symbols().clone(),
        array.symbol_lengths().clone(),
        take(array.codes().as_ref(), indices)?
            .as_::<VarBinVTable>()
            .clone(),
        fill_null(
            &take(array.uncompressed_lengths(), indices)?,
            &Scalar::new(
                array.uncompressed_lengths_dtype().clone(),
                ScalarValue::from(0),
            ),
        )?,
    )?
    .into_array())
}

impl TakeReduce for FSSTVTable {
    fn take(array: &FSSTArray, indices: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        take_fsst(array, indices).map(Some)
    }
}

impl FSSTVTable {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::take;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_take_null() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));
        let compr = fsst_train_compressor(&arr);
        let fsst = fsst_compress(&arr, &compr);

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            take(fsst.as_ref(), idx1.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            take(fsst.as_ref(), idx2.as_ref()).unwrap().dtype(),
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
        let compressor = fsst_train_compressor(&varbin);
        let array = fsst_compress(&varbin, &compressor);
        test_take_conformance(array.as_ref());
    }

    #[rstest]
    // Basic string arrays
    #[case::fsst_simple({
        let varbin = VarBinArray::from_iter(
            ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, &compressor)
    })]
    // Nullable strings
    #[case::fsst_nullable({
        let varbin = VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("test"), None],
            DType::Utf8(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(varbin, &compressor)
    })]
    // Repetitive patterns (good for FSST compression)
    #[case::fsst_repetitive({
        let varbin = VarBinArray::from_iter(
            ["http://example.com", "http://test.com", "http://vortex.dev", "http://data.org"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, &compressor)
    })]
    // Edge cases
    #[case::fsst_single({
        let varbin = VarBinArray::from_iter(
            ["single element"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(&varbin, &compressor)
    })]
    #[case::fsst_empty_strings({
        let varbin = VarBinArray::from_iter(
            ["", "test", "", "hello", ""].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(varbin, &compressor)
    })]
    // Large arrays
    #[case::fsst_large({
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
        fsst_compress(varbin, &compressor)
    })]

    fn test_fsst_consistency(#[case] array: FSSTArray) {
        test_array_consistency(array.as_ref());
    }
}
