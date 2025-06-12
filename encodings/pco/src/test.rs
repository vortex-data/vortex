use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Canonical, ToCanonical};
use vortex_buffer::Buffer;

use crate::PcoArray;

macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!($arr.scalar_at($n).unwrap(), $expected.try_into().unwrap());
    };
}

#[test]
fn test_pco_compress_decompress() {
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = PcoArray::from_primitive(&array, 3, 0).unwrap();
    // this data should be compressible
    assert!(compressed.pages.len() < array.nbytes());

    // check full decompression works
    let decompressed = compressed.decompress().unwrap().to_primitive().unwrap();
    assert_eq!(decompressed.as_slice::<i32>(), &data);

    // check slicing works
    let slice = compressed.slice(100, 105).unwrap();
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i);
    }
    match slice.to_canonical() {
        Ok(Canonical::Primitive(primitive)) => {
            assert_eq!(primitive.as_slice::<i32>(), &[100, 101, 102, 103, 104]);
        }
        _ => unreachable!(),
    }

    let slice = compressed.slice(200, 200).unwrap();
    match slice.to_canonical().unwrap() {
        Canonical::Primitive(primitive) => {
            assert_eq!(primitive.as_slice::<i32>(), &Vec::<i32>::new());
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_pco_empty() {
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = PcoArray::from_primitive(&array, 3, 100).unwrap();
    match compressed.decompress().unwrap().to_canonical().unwrap() {
        Canonical::Primitive(primitive) => {
            assert_eq!(primitive.as_slice::<i32>(), &data);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_pco_with_validity_and_multiple_chunks_and_pages() {
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![true; 200];
    validity[7..15].fill(false);
    validity[101] = false;
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(validity).to_array()),
    );
    let compression_level = 3;
    let values_per_chunk = 33;
    let values_per_page = 10;
    let compressed = PcoArray::from_primitive_with_values_per_chunk(
        &array,
        compression_level,
        values_per_chunk,
        values_per_page,
    )
    .unwrap();

    assert_eq!(compressed.metadata.chunks.len(), 6); // 191 values / 33 rounds up to 6
    assert_eq!(compressed.metadata.chunks[0].pages.len(), 4); // 33 / 10 rounds up to 4
    assert_nth_scalar!(compressed, 0, 0);
    assert_nth_scalar!(compressed, 3, 3);
    assert_nth_scalar!(compressed, 7, None::<i32>);
    assert_nth_scalar!(compressed, 14, None::<i32>);
    assert_nth_scalar!(compressed, 15, 15);
    assert_nth_scalar!(compressed, 101, None::<i32>);
    assert_nth_scalar!(compressed, 199, 199);

    // check slicing works
    let slice = compressed.slice(100, 103).unwrap();
    assert_nth_scalar!(slice, 0, 100);
    assert_nth_scalar!(slice, 2, 102);
    match slice.to_canonical() {
        Ok(Canonical::Primitive(primitive)) => {
            assert_eq!(
                primitive.validity(),
                &Validity::Array(BoolArray::from_iter(vec![true, false, true]).to_array())
            )
        }
        _ => unreachable!(),
    }
}
