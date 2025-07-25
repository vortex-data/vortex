// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;
use bytes::Bytes;
use itertools::Itertools;
use tokio::runtime::Handle;
use vortex::Array;
use vortex::file::{VortexLayoutStrategy, VortexOpenOptions, VortexWriteOptions};

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> anyhow::Result<u64> {
    Ok(VortexWriteOptions::default()
        .with_strategy(VortexLayoutStrategy::with_executor(Arc::new(
            Handle::current(),
        )))
        .write(Cursor::new(buf), array.to_array_stream())
        .await?
        .position())
}

#[inline(never)]
pub async fn vortex_decompress_read(buf: Bytes) -> anyhow::Result<Vec<RecordBatch>> {
    let scan = VortexOpenOptions::in_memory().open(buf)?.scan()?;
    let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

    Ok(scan
        .into_record_batch_reader_multithread(schema)?
        .try_collect()?)
}
