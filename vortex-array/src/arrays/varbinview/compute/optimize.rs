// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewVTable};
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::compute::{OptimizeKernel, OptimizeKernelAdapter};
use crate::{ArrayRef, register_kernel};

impl OptimizeKernel for VarBinViewVTable {
    fn optimize(&self, array: &VarBinViewArray) -> VortexResult<ArrayRef> {
        // If there are no buffers or only one buffer, no optimization needed
        if array.buffers().len() <= 1 {
            return Ok(array.to_array());
        }

        // Create a new builder and copy all elements
        let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());

        // Iterate through the array and append each element
        for i in 0..array.len() {
            if !array.is_valid(i)? {
                builder.append_null();
            } else {
                let bytes = array.bytes_at(i);
                builder.append_value(bytes.as_slice());
            }
        }

        Ok(builder.finish())
    }
}

register_kernel!(OptimizeKernelAdapter(VarBinViewVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::{VarBinViewArray, VarBinViewVTable};
    use crate::compute::{optimize, take};

    #[test]
    fn test_optimize_compacts_buffers() {
        // Create a VarBinViewArray with some long strings that will create multiple buffers
        let original = VarBinViewArray::from_iter_str([
            "short",
            "this is a longer string that will be stored in a buffer",
            "medium length string",
            "another very long string that definitely needs a buffer to store it",
            "tiny",
        ]);

        // Verify it has buffers
        assert!(original.nbuffers() > 0);
        let original_buffers = original.nbuffers();

        // Take only the first and last elements (indices 0 and 4)
        let indices = buffer![0u32, 4u32].into_array();
        let taken = take(original.as_ref(), &indices).unwrap();
        let taken_array = taken.as_::<VarBinViewVTable>();

        // The taken array should still have the same number of buffers
        assert_eq!(taken_array.nbuffers(), original_buffers);

        // Now optimize the taken array
        let optimized = optimize(taken_array.as_ref()).unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        // The optimized array should have compacted buffers
        // Since both remaining strings are short, they should be inlined
        // so we might have 0 buffers, or 1 buffer if any were not inlined
        assert!(optimized_array.nbuffers() <= 1);

        // Verify the data is still correct
        assert_eq!(optimized_array.len(), 2);
        assert_eq!(
            &*optimized_array
                .scalar_at(0)
                .unwrap()
                .as_utf8()
                .value()
                .unwrap(),
            "short"
        );
        assert_eq!(
            &*optimized_array
                .scalar_at(1)
                .unwrap()
                .as_utf8()
                .value()
                .unwrap(),
            "tiny"
        );
    }

    #[test]
    fn test_optimize_with_long_strings() {
        // Create strings that are definitely longer than 12 bytes
        let long_string_1 = "this is definitely a very long string that exceeds the inline limit";
        let long_string_2 = "another extremely long string that also needs external buffer storage";
        let long_string_3 = "yet another long string for testing buffer compaction functionality";

        let original = VarBinViewArray::from_iter_str([
            long_string_1,
            long_string_2,
            long_string_3,
            "short1",
            "short2",
        ]);

        // Take only the first and third long strings (indices 0 and 2)
        let indices = buffer![0u32, 2u32].into_array();
        let taken = take(original.as_ref(), &indices).unwrap();
        let taken_array = taken.as_::<VarBinViewVTable>();

        // Optimize the taken array
        let optimized = optimize(taken_array.as_ref()).unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        // The optimized array should have exactly 1 buffer (consolidated)
        assert_eq!(optimized_array.nbuffers(), 1);

        // Verify the data is still correct
        assert_eq!(optimized_array.len(), 2);
        assert_eq!(
            &*optimized_array
                .scalar_at(0)
                .unwrap()
                .as_utf8()
                .value()
                .unwrap(),
            long_string_1
        );
        assert_eq!(
            &*optimized_array
                .scalar_at(1)
                .unwrap()
                .as_utf8()
                .value()
                .unwrap(),
            long_string_3
        );
    }

    #[test]
    fn test_optimize_no_buffers() {
        // Create an array with only short strings (all inlined)
        let original = VarBinViewArray::from_iter_str(["a", "bb", "ccc", "dddd"]);

        // This should have no buffers
        assert_eq!(original.nbuffers(), 0);

        // Optimize should return the same array
        let optimized = optimize(original.as_ref()).unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        assert_eq!(optimized_array.nbuffers(), 0);
        assert_eq!(optimized_array.len(), 4);

        // Verify all values are preserved
        for i in 0..4 {
            assert_eq!(
                optimized_array.scalar_at(i).unwrap(),
                original.scalar_at(i).unwrap()
            );
        }
    }

    #[test]
    fn test_optimize_single_buffer() {
        // Create an array that naturally has only one buffer
        let original = VarBinViewArray::from_iter_str([
            "this is a long string that goes into a buffer",
            "another long string in the same buffer",
        ]);

        // Should have 1 buffer
        assert_eq!(original.nbuffers(), 1);

        // Optimize should return the same array (no change needed)
        let optimized = optimize(original.as_ref()).unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        assert_eq!(optimized_array.nbuffers(), 1);
        assert_eq!(optimized_array.len(), 2);

        // Verify all values are preserved
        for i in 0..2 {
            assert_eq!(
                optimized_array.scalar_at(i).unwrap(),
                original.scalar_at(i).unwrap()
            );
        }
    }
}
