use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::Buffer;

use crate::ZstdArray;

macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!($arr.scalar_at($n).unwrap(), $expected.try_into().unwrap());
    };
}

#[test]
fn test_zstd_compress_decompress() {
    let data: Vec<i32> = (0..1000).collect();
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );
    let array_ref = array.into_array();

    let compressed = ZstdArray::try_from_array_with_level(array_ref.clone(), 3).unwrap();
    // this data should be compressible
    assert!(compressed.compressed_data().len() < array_ref.nbytes());

    // check slicing works
    let slice = compressed.slice(100, 110).unwrap();
    for i in 0..10 {
        assert_nth_scalar!(slice, i, 100 + i as i32);
    }

    // check full decompression works
    let decompressed = compressed.decompress().unwrap().to_primitive().unwrap();
    assert_eq!(decompressed.as_slice::<i32>(), &data);
}
