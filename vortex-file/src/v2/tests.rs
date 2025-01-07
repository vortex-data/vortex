use bytes::Bytes;
use vortex_array::array::ChunkedArray;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ContextRef, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer;
use vortex_layout::scanner::Scan;

use crate::v2::{VortexOpenOptions, VortexWriteOptions};

#[tokio::test]
async fn write_read() {
    let arr = ChunkedArray::from_iter(vec![
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
    ])
    .into_array();

    let written: Bytes = VortexWriteOptions::default()
        .write(vec![], arr.into_array_stream())
        .await
        .unwrap()
        // TODO(ngates): no need to wrap Vec<u8> in Bytes if VortexReadAt doesn't require clone.
        .into();

    let vxf = VortexOpenOptions::new(ContextRef::default())
        .open(written)
        .await
        .unwrap();

    let result = vxf
        .scan(Scan::all())
        .unwrap()
        .into_array_data()
        .await
        .unwrap()
        .into_primitive()
        .unwrap();

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4, 5]);
}
