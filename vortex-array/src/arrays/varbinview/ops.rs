// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinViewArray, VarBinViewVTable, varbin_scalar};
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::validity::Validity;
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn slice(array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let views = array.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            array.buffers().to_vec(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }

    fn optimize(array: &VarBinViewArray) -> VortexResult<ArrayRef> {
        // If there is little to be gained by compacting, return the original array untouched.
        if !should_compact(array) {
            return Ok(array.to_array());
        }

        match array.validity {
            Validity::AllInvalid => {
                // The array contains no values, drop all buffers.
                Ok(VarBinViewArray::try_new(
                    array.views().clone(),
                    vec![],
                    array.dtype().clone(),
                    array.validity().clone(),
                )?
                .into_array())
            }
            // Non-null pathway
            Validity::NonNullable | Validity::AllValid => rebuild_nonnull(array),
            // Nullable pathway, requires null-checks for each value
            Validity::Array(_) => rebuild_nullable(array),
        }
    }
}

fn should_compact(array: &VarBinViewArray) -> bool {
    // If the array is entirely inlined strings, do not attempt to compact.
    if array.nbuffers() == 0 {
        return false;
    }

    // Scan the views, calculating the total buffer size that is referenced.
    let bytes_referenced: u64 = array
        .views()
        .iter()
        .map(|&view| {
            if view.is_inlined() {
                0u64
            } else {
                // SAFETY: in this branch the view is not inlined.
                unsafe { view._ref }.size as u64
            }
        })
        .sum();

    let buffer_total_size: u64 = array.buffers.iter().map(|buf| buf.len() as u64).sum();

    // If the majority of buffer space is unused, attempt to repack
    bytes_referenced < buffer_total_size / 2
}

// Nullable string array compaction pathway.
// This requires a null check on every append.
fn rebuild_nullable(array: &VarBinViewArray) -> VortexResult<ArrayRef> {
    let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());
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

// Compaction for string arrays that contain no null values. Saves a branch
// for every string element.
fn rebuild_nonnull(array: &VarBinViewArray) -> VortexResult<ArrayRef> {
    let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());
    for i in 0..array.len() {
        builder.append_value(array.bytes_at(i).as_ref());
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::{VarBinViewArray, VarBinViewVTable};
    use crate::compute::take;

    #[test]
    fn test_optimize_compacts_buffers() {
        // Create a VarBinViewArray with some long strings that will create multiple buffers
        let original = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            Some("this is a longer string that will be stored in a buffer"),
            Some("medium length string"),
            Some("another very long string that definitely needs a buffer to store it"),
            Some("tiny"),
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
        let optimized = taken_array.optimize().unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        // The optimized array should have compacted buffers
        // Since both remaining strings are short, they should be inlined
        // so we might have 0 buffers, or 1 buffer if any were not inlined
        assert!(optimized_array.nbuffers() <= 1);

        // Verify the data is still correct
        assert_eq!(optimized_array.len(), 2);
        assert_eq!(optimized_array.scalar_at(0).unwrap(), "short".into());
        assert_eq!(optimized_array.scalar_at(1).unwrap(), "tiny".into());
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
        let optimized = taken_array.optimize().unwrap();
        let optimized_array = optimized.as_::<VarBinViewVTable>();

        // The optimized array should have exactly 1 buffer (consolidated)
        assert_eq!(optimized_array.nbuffers(), 1);

        // Verify the data is still correct
        assert_eq!(optimized_array.len(), 2);
        assert_eq!(optimized_array.scalar_at(0).unwrap(), long_string_1.into());
        assert_eq!(optimized_array.scalar_at(1).unwrap(), long_string_3.into());
    }

    #[test]
    fn test_optimize_no_buffers() {
        // Create an array with only short strings (all inlined)
        let original = VarBinViewArray::from_iter_str(["a", "bb", "ccc", "dddd"]);

        // This should have no buffers
        assert_eq!(original.nbuffers(), 0);

        // Optimize should return the same array
        let optimized = original.optimize().unwrap();
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
        let optimized = original.optimize().unwrap();
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
