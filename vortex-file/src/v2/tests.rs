use bytes::Bytes;
use futures_executor::block_on;
use vortex_array::array::ChunkedArray;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ContextRef, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer;
use vortex_error::{VortexExpect, VortexResult};

use crate::v2::io::IoDriver;
use crate::v2::*;

fn chunked_file() -> VortexFile<impl IoDriver> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();

    block_on(async {
        let buffer: Bytes = VortexWriteOptions::default()
            .write(vec![], array.into_array_stream())
            .await?
            .into();
        VortexOpenOptions::new(ContextRef::default())
            .open(buffer)
            .await
    })
    .vortex_expect("Failed to create test file")
}

#[test]
fn basic_file_roundtrip() -> VortexResult<()> {
    let vxf = chunked_file();
    let result = block_on(vxf.scan(Scan::all())?.into_array_data())?.into_primitive()?;

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

    Ok(())
}

#[test]
fn file_take() -> VortexResult<()> {
    let vxf = chunked_file();
    let result = block_on(
        vxf.scan(Scan::all().with_row_indices(buffer![0, 1, 8]))?
            .into_array_data(),
    )?
    .into_primitive()?;

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 8]);

    Ok(())
}
