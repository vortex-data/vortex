//! Serialization and deserialization of a Vortex [`Array`].
//!
//! Arrays can be serialized into a single [`ByteBuffer`] that can be deserialized by providing
//! an encoding [`Context`], a [`DType`], and a row count.
//!
//! The serialized buffer can optionally be configured for zero-allocation reads by including
//! sufficient padding. If the buffer is to remain zero-allocation at its destination, you should
//! respect the returned [`ByteBuffer.alignment`] when allocating the destination buffer.

use vortex_buffer::ByteBuffer;

use crate::parts::ArrayParts;
use crate::Array;

pub struct SerializationOptions {
    /// Include padding to allow zero-allocation, zero-copy reads.
    pub include_padding: bool,
    // TODO(ngates): support compressing the flatbuffer metadata.
}

impl Array {
    pub fn serialize(self, _options: SerializationOptions) -> ByteBuffer {
        let _parts = ArrayParts::from(self);
        todo!()
    }
}
