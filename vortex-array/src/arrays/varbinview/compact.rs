//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defines a compaction operation for VarBinViewArrays that evicts unused buffers so they can
//! be dropped.

use vortex_error::{VortexResult, VortexUnwrap};

use crate::arrays::VarBinViewArray;
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// Returns a compacted copy of the input array, where all wasted space has been cleaned up. This
/// operation can be very expensive, in the worst cast copying all existing string data into
/// a new allocation.
///
/// After slicing/taking operations `VarBinViewArray`s can continue to hold references to buffers
/// that are no longer visible. We detect when there is wasted space in any of the buffers, and if
/// so, will aggressively compact all visile outlined string data into a single new buffer.
pub fn compact_buffers(array: &VarBinViewArray) -> VortexResult<VarBinViewArray> {
    // If there is nothing to be gained by compaction, return the original array untouched.
    if !should_compact(array) {
        return Ok(array.clone());
    }

    // Compaction pathways, depend on the validity
    match array.validity() {
        // The array contains no values, all buffers can be dropped.
        Validity::AllInvalid => Ok(VarBinViewArray::try_new(
            array.views().clone(),
            vec![],
            array.dtype().clone(),
            array.validity().clone(),
        )?),
        // Non-null pathway
        Validity::NonNullable | Validity::AllValid => rebuild_nonnull(array),
        // Nullable pathway, requires null-checks for each value
        Validity::Array(_) => rebuild_nullable(array),
    }
}

fn should_compact(array: &VarBinViewArray) -> bool {
    // If the array is entirely inlined strings, do not attempt to compact.
    if array.nbuffers() == 0 {
        return false;
    }

    let bytes_referenced: u64 = count_referenced_bytes(array);
    let buffer_total_bytes: u64 = array.buffers.iter().map(|buf| buf.len() as u64).sum();

    // If there is any wasted space, we want to repack.
    // This is very aggressive.
    bytes_referenced < buffer_total_bytes
}

// count the number of bytes addressed by the views, not including null
// values or any inlined strings.
fn count_referenced_bytes(array: &VarBinViewArray) -> u64 {
    match array.validity() {
        Validity::AllInvalid => 0u64,
        _ => {
            array
                .views()
                .iter()
                .enumerate()
                .map(|(idx, &view)| {
                    if !array.is_valid(idx).vortex_unwrap() || view.is_inlined() {
                        0u64
                    } else {
                        // SAFETY: in this branch the view is not inlined.
                        unsafe { view._ref }.size as u64
                    }
                })
                .sum()
        }
    }
}

// Nullable string array compaction pathway.
// This requires a null check on every append.
fn rebuild_nullable(array: &VarBinViewArray) -> VortexResult<VarBinViewArray> {
    let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());
    for i in 0..array.len() {
        if !array.is_valid(i)? {
            builder.append_null();
        } else {
            let bytes = array.bytes_at(i);
            builder.append_value(bytes.as_slice());
        }
    }

    Ok(builder.finish_into_varbinview())
}

// Compaction for string arrays that contain no null values. Saves a branch
// for every string element.
fn rebuild_nonnull(array: &VarBinViewArray) -> VortexResult<VarBinViewArray> {
    let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());
    for i in 0..array.len() {
        builder.append_value(array.bytes_at(i).as_ref());
    }
    Ok(builder.finish_into_varbinview())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::varbinview::compact::compact_buffers;
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
        let optimized_array = compact_buffers(taken_array).unwrap();

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
        let optimized_array = compact_buffers(taken_array).unwrap();

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
        let optimized_array = compact_buffers(&original).unwrap();

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
        let str1 = "this is a long string that goes into a buffer";
        let str2 = "another long string in the same buffer";
        let original = VarBinViewArray::from_iter_str([str1, str2]);

        // Should have 1 compact buffer
        assert_eq!(original.nbuffers(), 1);
        assert_eq!(original.buffer(0).len(), str1.len() + str2.len());

        // Optimize should return the same array (no change needed)
        let optimized_array = compact_buffers(&original).unwrap();

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
