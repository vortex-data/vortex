// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use vortex::Array;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> anyhow::Result<u64> {
    let mut cursor = Cursor::new(buf);
    VortexWriteOptions::default()
        .write(&mut cursor, array.to_array_stream())
        .await?;
    Ok(cursor.position())
}

#[inline(never)]
pub async fn vortex_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let scan = VortexOpenOptions::new().open_buffer(buf)?.scan()?;
    let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

    let iter = scan.into_record_batch_reader_multithread(schema)?;
    let mut nbytes = 0;
    for batch in iter {
        nbytes += batch.unwrap().get_array_memory_size()
    }
    Ok(nbytes)
}
