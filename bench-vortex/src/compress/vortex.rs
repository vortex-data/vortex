// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use tokio::runtime::Handle;
use vortex::Array;
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> anyhow::Result<u64> {
    Ok(VortexWriteOptions::default()
        .with_strategy(
            WriteStrategyBuilder::new()
                .with_executor(Arc::new(Handle::current()))
                .build(),
        )
        .write(Cursor::new(buf), array.to_array_stream())
        .await?
        .position())
}

#[inline(never)]
pub async fn vortex_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let scan = VortexOpenOptions::in_memory().open(buf)?.scan()?;
    let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

    let iter = scan.into_record_batch_reader_multithread(schema)?;
    let mut nbytes = 0;
    for batch in iter {
        nbytes += batch.unwrap().get_array_memory_size()
    }
    Ok(nbytes)
}
