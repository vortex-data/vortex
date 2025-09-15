// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;

use bytes::Bytes;
use futures::{StreamExt, pin_mut};
use vortex::error::VortexExpect;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::{Array, IntoArray};

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

    let stream = scan.map(|a| Ok(a.to_canonical())).into_stream()?;
    pin_mut!(stream);

    let mut nbytes = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        nbytes += batch.into_array().nbytes();
    }
    Ok(nbytes.try_into().vortex_expect("nbytes overflow"))
}
