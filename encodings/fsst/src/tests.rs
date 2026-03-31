// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use crate::FSST;
use crate::fsst_compress;
use crate::fsst_train_compressor;

/// this function is VERY slow on miri, so we only want to run it once
pub(crate) fn build_fsst_array() -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.append_value(b"this compression ratio slaps hard ngl it really hits different");
    input_array.append_value(
        b"deadass vortex string handling is unmatched it really cleared the competition with zero effort",
    );
    input_array.append_value(b"touch grass if you think parquet is better fax no printer on god");
    let input_array = input_array.finish(DType::Utf8(Nullability::NonNullable));

    let compressor = fsst_train_compressor(&input_array);
    fsst_compress(input_array, &compressor).into_array()
}

#[test]
fn test_fsst_array_ops() {
    // first test the scalar_at values
    let fsst_array = build_fsst_array();
    assert_nth_scalar!(
        fsst_array,
        0,
        "this compression ratio slaps hard ngl it really hits different"
    );
    assert_nth_scalar!(
        fsst_array,
        1,
        "deadass vortex string handling is unmatched it really cleared the competition with zero effort"
    );
    assert_nth_scalar!(
        fsst_array,
        2,
        "touch grass if you think parquet is better fax no printer on god"
    );

    // test slice
    let fsst_sliced = fsst_array.slice(1..3).unwrap();
    assert!(fsst_sliced.is::<FSST>());
    assert_eq!(fsst_sliced.len(), 2);
    assert_nth_scalar!(
        fsst_sliced,
        0,
        "deadass vortex string handling is unmatched it really cleared the competition with zero effort"
    );
    assert_nth_scalar!(
        fsst_sliced,
        1,
        "touch grass if you think parquet is better fax no printer on god"
    );

    // test take
    let indices = buffer![0, 2].into_array();
    let fsst_taken = fsst_array.take(indices).unwrap();
    assert_eq!(fsst_taken.len(), 2);
    assert_nth_scalar!(
        fsst_taken,
        0,
        "this compression ratio slaps hard ngl it really hits different"
    );
    assert_nth_scalar!(
        fsst_taken,
        1,
        "touch grass if you think parquet is better fax no printer on god"
    );

    // test filter
    let mask = Mask::from_iter([false, true, true]);

    let fsst_filtered = fsst_array.filter(mask).unwrap();

    assert_eq!(fsst_filtered.len(), 2);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "deadass vortex string handling is unmatched it really cleared the competition with zero effort"
    );

    // test to_canonical
    let canonical_array = fsst_array.to_varbinview().into_array();

    assert_arrays_eq!(fsst_array.to_array(), canonical_array);
}
