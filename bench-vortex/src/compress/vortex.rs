// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use futures::pin_mut;
use vortex::array::Array;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;

use crate::SESSION;

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> anyhow::Result<u64> {
    let mut cursor = Cursor::new(buf);
    SESSION
        .write_options()
        .write(&mut cursor, array.to_array_stream())
        .await?;
    Ok(cursor.position())
}

#[inline(never)]
pub async fn vortex_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let scan = SESSION.open_options().open_buffer(buf)?.scan()?;
    let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

    let stream = scan.into_record_batch_stream(schema)?;
    pin_mut!(stream);

    let mut nbytes = 0;
    while let Some(batch) = stream.next().await {
        nbytes += batch?.get_array_memory_size()
    }
    Ok(nbytes)
}
