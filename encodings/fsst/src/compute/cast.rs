// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBin;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;
impl CastReduce for FSST {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // FSST is a string compression encoding.
        // For nullability changes, we can cast the codes and symbols arrays
        if array.dtype().eq_ignore_nullability(dtype) {
            // Cast codes array to handle nullability
            let new_codes = array
                .codes()
                .into_array()
                .cast(array.codes_dtype().with_nullability(dtype.nullability()))?;

            Ok(Some(
                FSST::try_new(
                    dtype.clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    new_codes.as_::<VarBin>().into_owned(),
                    array.uncompressed_lengths().clone(),
                )?
                .into_array(),
            ))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_cast_fsst_nullability() {
        let strings = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), Some("hello world")],
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = fsst_train_compressor(&strings);
        let len = strings.len();
        let dtype = strings.dtype().clone();
        let fsst = fsst_compress(strings, len, &dtype, &compressor);

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
        let compressor = fsst_train_compressor(&array);
        let fsst = fsst_compress(&array, array.len(), array.dtype(), &compressor);
        test_cast_conformance(&fsst.into_array());
    }
}
