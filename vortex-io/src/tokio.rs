// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;

use tokio::io::AsyncWriteExt;

use crate::{IoBuf, VortexWrite};

impl VortexWrite for tokio::fs::File {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        AsyncWriteExt::flush(self).await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        AsyncWriteExt::shutdown(self).await
    }
}
