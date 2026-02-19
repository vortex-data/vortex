// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterKernel;
use vortex_array::arrays::VarBinVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FSSTArray;
use crate::FSSTVTable;

impl FilterKernel for FSSTVTable {
    fn filter(
        array: &FSSTArray,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Directly invoke VarBin's FilterKernel to get a concrete VarBinArray back.
        let filtered_codes = <VarBinVTable as FilterKernel>::filter(array.codes(), mask, ctx)?
            .vortex_expect("VarBin filter kernel always returns Some")
            .as_::<VarBinVTable>()
            .clone();

        Ok(Some(
            FSSTArray::try_new(
                array.dtype().clone(),
                array.symbols().clone(),
                array.symbol_lengths().clone(),
                filtered_codes,
                array.uncompressed_lengths().filter(mask.clone())?,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::builder::VarBinBuilder;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_filter_fsst_array() {
        // Test with small strings
        let mut builder = VarBinBuilder::<i32>::with_capacity(5);
        builder.append_value(b"hello");
        builder.append_value(b"world");
        builder.append_value(b"hello");
        builder.append_value(b"rust");
        builder.append_value(b"world");
        let varbin = builder.finish(DType::Utf8(Nullability::NonNullable));

        let compressor = fsst_train_compressor(&varbin);
        let array = fsst_compress(&varbin, &compressor);
        test_filter_conformance(array.as_ref());

        // Test with longer strings that benefit from compression
        let mut builder = VarBinBuilder::<i32>::with_capacity(5);
        builder.append_value(b"the quick brown fox");
        builder.append_value(b"the quick brown fox jumps");
        builder.append_value(b"the lazy dog");
        builder.append_value(b"the quick brown fox jumps over");
        builder.append_value(b"the lazy dog sleeps");
        let varbin = builder.finish(DType::Utf8(Nullability::NonNullable));

        let compressor = fsst_train_compressor(&varbin);
        let array = fsst_compress(&varbin, &compressor);
        test_filter_conformance(array.as_ref());

        // Test with nullable strings
        let mut builder = VarBinBuilder::<i32>::with_capacity(5);
        builder.append_value(b"compress");
        builder.append_null();
        builder.append_value(b"decompress");
        builder.append_value(b"compress");
        builder.append_null();
        let varbin = builder.finish(DType::Utf8(Nullability::Nullable));

        let compressor = fsst_train_compressor(&varbin);
        let array = fsst_compress(&varbin, &compressor);
        test_filter_conformance(array.as_ref());
    }
}
