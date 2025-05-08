use vortex_array::arrays::builder::VarBinBuilder;
use vortex_array::compute::{filter, take};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability};
use vortex_mask::Mask;

use crate::{FSSTEncoding, fsst_compress, fsst_train_compressor};

macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!($arr.scalar_at($n).unwrap(), $expected.try_into().unwrap());
    };
}

// this function is VERY slow on miri, so we only want to run it once
fn build_fsst_array() -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.append_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.append_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.append_value(b"Nothing in present history can contradict them");
    let input_array = input_array.finish(DType::Utf8(Nullability::NonNullable));

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
    let fsst_sliced = fsst_array.slice(1, 3).unwrap();
    assert_eq!(fsst_sliced.encoding(), FSSTEncoding.id());
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
    assert_eq!(fsst_taken.encoding(), FSSTEncoding.id());
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
    let mask = Mask::from_iter([false, true, true]);

    let fsst_filtered = filter(&fsst_array, &mask).unwrap();
    assert_eq!(fsst_filtered.encoding(), FSSTEncoding.id());
    assert_eq!(fsst_filtered.len(), 2);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );

    // test to_canonical
    let canonical_array = fsst_array.to_varbinview().unwrap().into_array();

    assert_eq!(canonical_array.len(), fsst_array.len());

    for i in 0..fsst_array.len() {
        assert_eq!(
            fsst_array.scalar_at(i).unwrap(),
            canonical_array.scalar_at(i).unwrap(),
        );
    }
}
