// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffer;
#[cfg(feature = "object_store")]
pub(crate) mod object_store;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// A blocking write interface for I/O operations.
pub trait Write {
    fn write(&mut self, buffer: ByteBuffer) -> VortexResult<ByteBuffer>;
    fn flush(&mut self) -> VortexResult<()>;
}

/// A Tokio-specific asynchronous write interface for I/O operations.
///
/// Passing this trait assumes the returned future will be awaited within a Tokio runtime context.
pub trait TokioWrite {
    fn write(&mut self, buffer: ByteBuffer) -> impl Future<Output = VortexResult<ByteBuffer>>;
    fn flush(&mut self) -> impl Future<Output = VortexResult<()>>;
    fn shutdown(&mut self) -> impl Future<Output = VortexResult<()>>;
}
