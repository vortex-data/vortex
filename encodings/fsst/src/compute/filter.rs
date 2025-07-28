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
    use vortex_array::compute::conformance::filter::test_filter_conformance;

    use crate::FSSTArray;

    #[test]
    fn test_filter_fsst_array() {
        // Test with small strings
        let strings = vec!["hello", "world", "hello", "rust", "world"];
        let array = FSSTArray::from_iter(strings.iter().map(|s| Some(*s))).unwrap();
        test_filter_conformance(array.as_ref());

        // Test with longer strings that benefit from compression
        let strings = vec![
            "the quick brown fox",
            "the quick brown fox jumps",
            "the lazy dog",
            "the quick brown fox jumps over",
            "the lazy dog sleeps",
        ];
        let array = FSSTArray::from_iter(strings.iter().map(|s| Some(*s))).unwrap();
        test_filter_conformance(array.as_ref());

        // Test with nullable strings
        let strings = vec![
            Some("compress"),
            None,
            Some("decompress"),
            Some("compress"),
            None,
        ];
        let array = FSSTArray::from_iter(strings).unwrap();
        test_filter_conformance(array.as_ref());
    }
}
