// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::VortexResult;

/// A trait for reading fixed byte ranges from underlying I/O.
///
/// While this trait uses async/await syntax, it is intended to be used within the Vortex CPU
/// execution model, and therefore must be runtime agnostic. Any underlying I/O that requires
/// a specific runtime, is `!Send`, or has other specific requirements should be dispatched using
/// one of the provided runtime dispatchers.
///
/// For this reason, the trait is sealed. For providing custom implementations, you are encouraged
/// to implement the trait required for a specific runtime dispatcher.
#[async_trait]
pub trait ReadAt: 'static + Send + Sync + private::Sealed {
    /// Read the byte range specified by `offset` and `len` into a new `ByteBuffer` with the
    /// requested `alignment`.
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer>;

    async fn size(&self) -> VortexResult<u64>;
}

mod private {
    use vortex_buffer::ByteBuffer;

    use crate::tokio::{TokioDispatchedIo, TokioReadAt};

    pub trait Sealed {}

    impl Sealed for ByteBuffer {}
    impl<R: TokioReadAt> Sealed for TokioDispatchedIo<R> {}
}
