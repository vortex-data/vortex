// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits for reading and writing Vortex types as flatbuffers.

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::InvalidFlatbuffer;
use flatbuffers::Verifiable;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ConstByteBuffer;
use vortex_error::VortexResult;

/// We define a const-aligned byte buffer for flatbuffers with 8-byte alignment.
///
/// This is based on the assumption that the maximum primitive type is 8 bytes.
/// See: <https://groups.google.com/g/flatbuffers/c/PSgQeWeTx_g>
pub type FlatBuffer = ConstByteBuffer<8>;

/// Marker trait for types that can be the root of a FlatBuffer.
pub trait FlatBufferRoot {}

/// Trait for reading a type from a FlatBuffer.
pub trait ReadFlatBuffer: Sized {
    /// The FlatBuffer type that this type can be read from.
    type Source<'a>: Verifiable + Follow<'a>;
    /// The error type returned when reading fails.
    type Error: From<InvalidFlatbuffer>;

    /// Reads this type from a FlatBuffer source.
    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error>;

    /// Reads this type from bytes representing a FlatBuffer source.
    fn read_flatbuffer_bytes<'buf>(bytes: &'buf [u8]) -> Result<Self, Self::Error>
    where
        <Self as ReadFlatBuffer>::Source<'buf>: 'buf,
    {
        let fb = root::<Self::Source<'buf>>(bytes)?;
        Self::read_flatbuffer(&fb)
    }
}

/// Trait for writing a type to a FlatBuffer.
pub trait WriteFlatBuffer {
    /// The FlatBuffer type that this type can be written to.
    type Target<'a>;

    /// Writes this type to a FlatBuffer builder.
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>>;
}

/// Extension trait for types that can be written as FlatBuffer root objects.
pub trait WriteFlatBufferExt: WriteFlatBuffer + FlatBufferRoot {
    /// Writes self as a FlatBuffer root object into a [`FlatBuffer`] byte buffer.
    fn write_flatbuffer_bytes(&self) -> VortexResult<FlatBuffer>;
}

impl<F: WriteFlatBuffer + FlatBufferRoot> WriteFlatBufferExt for F {
    fn write_flatbuffer_bytes(&self) -> VortexResult<FlatBuffer> {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = self.write_flatbuffer(&mut fbb)?;
        fbb.finish_minimal(root_offset);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        Ok(FlatBuffer::align_from(
            ByteBuffer::from(vec).slice(start..end),
        ))
    }
}
