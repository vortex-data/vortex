// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{FSSTArray, FSSTVTable};

impl CastKernel for FSSTVTable {
    fn cast(&self, array: &FSSTArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // FSST is a string compression encoding.
        // For nullability changes, we can cast the codes and symbols arrays
        if array.dtype().eq_ignore_nullability(dtype) {
            // Cast codes array to handle nullability
            let new_codes = cast(
                array.codes().as_ref(),
                &array.codes().dtype().with_nullability(dtype.nullability()),
            )?;

            Ok(Some(
                FSSTArray::try_new(
                    dtype.clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    new_codes.as_::<VarBinVTable>().clone(),
                    array.uncompressed_lengths().clone(),
                )?
                .into_array(),
            ))
        } else {
            Ok(None)
        }
    }
}

register_kernel!(CastKernelAdapter(FSSTVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_dtype::{DType, Nullability};

    use crate::{fsst_compress, fsst_train_compressor};

    #[test]
    fn test_cast_fsst_nullability() {
        let strings = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), Some("hello world")],
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = fsst_train_compressor(&strings);
        let fsst = fsst_compress(strings, &compressor);

        // Cast to nullable
        let casted = cast(fsst.as_ref(), &DType::Utf8(Nullability::Nullable)).unwrap();
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
        let fsst = fsst_compress(&array, &compressor);
        test_cast_conformance(fsst.as_ref());
    }
}
