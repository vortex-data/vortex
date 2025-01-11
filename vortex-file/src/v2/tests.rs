use bytes::Bytes;
use futures_executor::block_on;
use vortex_array::array::ChunkedArray;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ContextRef, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_scan::Scan;

use crate::v2::*;

#[test]
fn basic_file_roundtrip() -> VortexResult<()> {
    block_on(async {
        let array = ChunkedArray::from_iter([
            buffer![0, 1, 2].into_array(),
            buffer![3, 4, 5].into_array(),
            buffer![6, 7, 8].into_array(),
        ])
        .into_array();

        let buffer: Bytes = VortexWriteOptions::default()
            .write_async(vec![], array.into_array_stream())
            .await?
            .into();

        let vxf = VortexOpenOptions::new(ContextRef::default())
            .open(buffer)
            .await?;
        let result = vxf
            .scan(Scan::all())?
            .into_array_data()
            .await?
            .into_primitive()?;

        assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

        Ok(())
    })
}
