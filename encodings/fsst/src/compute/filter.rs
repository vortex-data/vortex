// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::{FilterKernel, FilterKernelAdapter, filter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{FSSTArray, FSSTVTable};

impl FilterKernel for FSSTVTable {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, array: &FSSTArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            filter(array.codes().as_ref(), mask)?
                .as_::<VarBinVTable>()
                .clone(),
            filter(array.uncompressed_lengths(), mask)?,
        )?
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(FSSTVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::arrays::builder::VarBinBuilder;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_dtype::{DType, Nullability};

    use crate::{fsst_compress, fsst_train_compressor};

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
