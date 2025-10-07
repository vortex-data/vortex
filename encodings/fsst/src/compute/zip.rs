// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::{ZipKernel, ZipKernelAdapter, zip};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{FSSTArray, FSSTVTable};

impl ZipKernel for FSSTVTable {
    fn zip(
        &self,
        if_true: &FSSTArray,
        if_false: &dyn Array,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        // If if_false is also an FSST array with the same symbol table, we can zip efficiently
        if let Some(if_false_fsst) = if_false.as_opt::<FSSTVTable>() {
            // Only proceed if both arrays share the same symbol table
            if if_true.symbols().iter().zip(if_false_fsst.symbols().iter()).all(|(l, r)| l.) == 
                && if_true.symbol_lengths() == if_false_fsst.symbol_lengths()
            {
                return Ok(Some(
                    FSSTArray::try_new(
                        if_true.dtype().clone(),
                        if_true.symbols().clone(),
                        if_true.symbol_lengths().clone(),
                        zip(
                            if_true.codes().as_ref(),
                            if_false_fsst.codes().as_ref(),
                            mask,
                        )?
                        .as_::<VarBinVTable>()
                        .clone(),
                        zip(
                            if_true.uncompressed_lengths(),
                            if_false_fsst.uncompressed_lengths(),
                            mask,
                        )?,
                    )?
                    .into_array(),
                ));
            }
        }

        // If symbol tables don't match, fall back to canonical
        Ok(None)
    }
}

register_kernel!(ZipKernelAdapter(FSSTVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::zip;
    use vortex_dtype::{DType, Nullability};
    use vortex_mask::Mask;

    use crate::{fsst_compress, fsst_train_compressor};

    #[test]
    fn test_zip_same_compressor() {
        let arr1 = VarBinArray::from_iter(
            ["hello", "world", "test", "data", "vortex"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let arr2 = VarBinArray::from_iter(
            ["foo", "bar", "baz", "qux", "quux"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );

        // Use the same compressor for both arrays
        let compressor = fsst_train_compressor(arr1.as_ref()).unwrap();
        let fsst1 = fsst_compress(arr1.as_ref(), &compressor).unwrap();
        let fsst2 = fsst_compress(arr2.as_ref(), &compressor).unwrap();

        let mask = Mask::from_iter([true, false, true, false, true]);
        let result = zip(fsst1.as_ref(), fsst2.as_ref(), &mask).unwrap();

        test_array_consistency(result.as_ref());
    }

    #[test]
    fn test_zip_different_compressor() {
        let arr1 = VarBinArray::from_iter(
            ["hello", "world", "test"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );
        let arr2 = VarBinArray::from_iter(
            ["foo", "bar", "baz"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        );

        // Use different compressors
        let compressor1 = fsst_train_compressor(arr1.as_ref()).unwrap();
        let compressor2 = fsst_train_compressor(arr2.as_ref()).unwrap();
        let fsst1 = fsst_compress(arr1.as_ref(), &compressor1).unwrap();
        let fsst2 = fsst_compress(arr2.as_ref(), &compressor2).unwrap();

        let mask = Mask::from_iter([true, false, true]);
        let result = zip(fsst1.as_ref(), fsst2.as_ref(), &mask).unwrap();

        test_array_consistency(result.as_ref());
    }

    #[test]
    fn test_zip_nullable() {
        let arr1 = VarBinArray::from_iter(
            [Some("hello"), None, Some("test")],
            DType::Utf8(Nullability::Nullable),
        );
        let arr2 = VarBinArray::from_iter(
            [Some("foo"), Some("bar"), None],
            DType::Utf8(Nullability::Nullable),
        );

        let compressor = fsst_train_compressor(arr1.as_ref()).unwrap();
        let fsst1 = fsst_compress(arr1.as_ref(), &compressor).unwrap();
        let fsst2 = fsst_compress(arr2.as_ref(), &compressor).unwrap();

        let mask = Mask::from_iter([true, false, true]);
        let result = zip(fsst1.as_ref(), fsst2.as_ref(), &mask).unwrap();

        test_array_consistency(result.as_ref());
    }
}
