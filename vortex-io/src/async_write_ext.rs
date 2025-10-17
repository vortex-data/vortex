// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::AsyncWriteExt;

use crate::{IoBuf, VortexWrite};

impl<T> VortexWrite for T
where
    T: AsyncWriteExt + Unpin,
{
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = std::io::Result<B>> {
        Box::pin(async move {
            AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
            Ok(buffer)
        })
    }

    fn flush(&mut self) -> impl Future<Output = std::io::Result<()>> {
        AsyncWriteExt::flush(self)
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> {
        AsyncWriteExt::close(self)
    }
}
