use std::io::Cursor;

use arrow_array::ArrayRef;
use bytes::Bytes;
use futures::TryStreamExt;
use tokio::runtime::Handle;
use vortex::Array;
use vortex::arrow::IntoArrowArray;
use vortex::error::VortexResult;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::stream::ArrayStreamArrayExt;

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> VortexResult<u64> {
    Ok(VortexWriteOptions::default()
        .write(Cursor::new(buf), array.to_array_stream())
        .await?
        .position())
}

#[inline(never)]
pub async fn vortex_decompress_read(buf: Bytes) -> VortexResult<Vec<ArrayRef>> {
    VortexOpenOptions::in_memory()
        .open(buf)
        .await?
        .scan()?
        .with_tokio_executor(Handle::current())
        .into_array_stream()?
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .map(|a| a.into_arrow_preferred())
        .collect::<VortexResult<Vec<_>>>()
}
