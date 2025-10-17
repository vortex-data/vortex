// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "futures")]
impl<T> crate::VortexWrite for T
where
    T: futures::AsyncWriteExt + Unpin,
{
    fn write_all<B: crate::IoBuf>(
        &mut self,
        buffer: B,
    ) -> impl Future<Output = std::io::Result<B>> {
        Box::pin(async move {
            futures::AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
            Ok(buffer)
        })
    }

    fn flush(&mut self) -> impl Future<Output = std::io::Result<()>> {
        futures::AsyncWriteExt::flush(self)
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> {
        futures::AsyncWriteExt::close(self)
    }
}

#[cfg(all(feature = "tokio", not(feature = "futures")))]
impl<T> crate::VortexWrite for T
where
    T: tokio::io::AsyncWriteExt + Unpin,
{
    fn write_all<B: crate::IoBuf>(
        &mut self,
        buffer: B,
    ) -> impl Future<Output = std::io::Result<B>> {
        Box::pin(async move {
            tokio::io::AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
            Ok(buffer)
        })
    }

    fn flush(&mut self) -> impl Future<Output = std::io::Result<()>> {
        tokio::io::AsyncWriteExt::flush(self)
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> {
        tokio::io::AsyncWriteExt::shutdown(self)
    }
}
