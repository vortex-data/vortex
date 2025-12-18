// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::pin_mut;
use vortex::array::Array;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex_bench::Format;
use vortex_bench::SESSION;

use crate::bench::Compressor;

/// Compressor implementation for Vortex format.
pub struct VortexCompressor;

#[async_trait]
impl Compressor for VortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn compress(&self, array: &dyn Array) -> Result<(Bytes, Duration)> {
        let mut buf = Vec::new();
        let start = Instant::now();
        vortex_compress_write(array, &mut buf).await?;
        let elapsed = start.elapsed();
        Ok((Bytes::from(buf), elapsed))
    }

    async fn decompress(&self, data: Bytes) -> Result<usize> {
        let scan = SESSION.open_options().open_buffer(data)?.scan()?;
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

        let stream = scan.into_record_batch_stream(schema)?;
        pin_mut!(stream);

        let mut nbytes = 0;
        while let Some(batch) = stream.next().await {
            nbytes += batch?.get_array_memory_size()
        }
        Ok(nbytes)
    }
}

#[inline(never)]
pub async fn vortex_compress_write(array: &dyn Array, buf: &mut Vec<u8>) -> Result<u64> {
    let mut cursor = Cursor::new(buf);
    SESSION
        .write_options()
        .write(&mut cursor, array.to_array_stream())
        .await?;
    Ok(cursor.position())
}
