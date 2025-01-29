#![cfg(test)]

use vortex_array::array::builder::VarBinBuilder;
use vortex_array::compute::{filter, scalar_at, slice, take};
use vortex_array::encoding::Encoding;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability};
use vortex_fsst::{fsst_compress, fsst_train_compressor, FSSTEncoding};
use vortex_mask::Mask;

macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!(scalar_at(&$arr, $n).unwrap(), $expected.try_into().unwrap());
    };
}

// this function is VERY slow on miri, so we only want to run it once
fn build_fsst_array() -> ArrayData {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.push_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.push_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.push_value(b"Nothing in present history can contradict them");
    let input_array = input_array
        .finish(DType::Utf8(Nullability::NonNullable))
        .into_array();

    let compressor = fsst_train_compressor(&input_array).unwrap();

    fsst_compress(&input_array, &compressor)
        .unwrap()
        .into_array()
}

#[test]
fn test_fsst_array_ops() {
    // first test the scalar_at values
    let fsst_array = build_fsst_array();
    assert_nth_scalar!(
        fsst_array,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_array,
        1,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_array,
        2,
        "Nothing in present history can contradict them"
    );

    // test slice
    let fsst_sliced = slice(&fsst_array, 1, 3).unwrap();
    assert_eq!(fsst_sliced.encoding().id(), FSSTEncoding::ID);
    assert_eq!(fsst_sliced.len(), 2);
    assert_nth_scalar!(
        fsst_sliced,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_sliced,
        1,
        "Nothing in present history can contradict them"
    );

    // test take
    let indices = buffer![0, 2].into_array();
    let fsst_taken = take(&fsst_array, &indices).unwrap();
    assert_eq!(fsst_taken.len(), 2);
    assert_nth_scalar!(
        fsst_taken,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_taken,
        1,
        "Nothing in present history can contradict them"
    );

    // test filter
    let mask = Mask::from_iter([false, true, false]);

    let fsst_filtered = filter(&fsst_array, &mask).unwrap();
    assert_eq!(fsst_filtered.encoding().id(), FSSTEncoding::ID);
    assert_eq!(fsst_filtered.len(), 1);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );

    // test into_canonical
    let canonical_array = fsst_array.clone().into_varbinview().unwrap().into_array();

    assert_eq!(canonical_array.len(), fsst_array.len());

    for i in 0..fsst_array.len() {
        assert_eq!(
            scalar_at(&fsst_array, i).unwrap(),
            scalar_at(&canonical_array, i).unwrap(),
        );
    }
}
