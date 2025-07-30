// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::io::Write;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::VortexResult;

impl Write for ByteBufferMut {
    fn write(&mut self, buffer: ByteBuffer) -> VortexResult<ByteBuffer> {
        self.extend_from_slice(buffer.as_slice());
        Ok(buffer)
    }

    fn flush(&mut self) -> VortexResult<()> {
        Ok(())
    }
}
