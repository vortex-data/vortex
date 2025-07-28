// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatch;
use itertools::Itertools;
use vortex::Array;
use vortex::buffer::{ByteBuffer, ByteBufferMut};
use vortex::file::{VortexLayoutStrategy, VortexOpenOptions, VortexWriteOptions};
use vortex::layout::LocalExecutor;

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array) -> anyhow::Result<ByteBuffer> {
    Ok(VortexWriteOptions::default()
        .with_strategy(VortexLayoutStrategy::with_executor(Arc::new(LocalExecutor)))
        .write(ByteBufferMut::empty(), array.to_array_stream())
        .await?
        .freeze())
}

#[inline(never)]
pub fn vortex_decompress_read(buf: ByteBuffer) -> anyhow::Result<Vec<RecordBatch>> {
    let scan = VortexOpenOptions::new(buf).open()?.scan()?;
    let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

    Ok(scan
        .into_record_batch_reader_multithread(schema)?
        .try_collect()?)
}
