// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A contiguously serialized Vortex array.
//!
//! See the `vortex-file` crate for non-contiguous serialization.

#![deny(missing_docs)]

#[cfg(feature = "array")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::unwrap_in_result)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[allow(missing_docs)]
#[rustfmt::skip]
#[path = "./generated/array.rs"]
/// A serialized array without its buffer (i.e. data).
///
/// `array.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-array/array.fbs")]
/// ```
pub mod array;

#[cfg(feature = "dtype")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[allow(missing_docs)]
#[rustfmt::skip]
#[path = "./generated/dtype.rs"]
/// A serialized data type.
///
/// `dtype.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-dtype/dtype.fbs")]
/// ```
pub mod dtype;

#[cfg(feature = "file")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::unwrap_in_result)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[allow(missing_docs)]
#[rustfmt::skip]
#[path = "./generated/footer.rs"]
/// A file format footer containing a serialized `vortex-file` Layout.
///
/// `footer.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-file/footer.fbs")]
/// ```
pub mod footer;

#[cfg(feature = "layout")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[allow(missing_docs)]
#[rustfmt::skip]
#[path = "./generated/layout.rs"]
/// Structures describing the physical layout of Vortex arrays in random access storage.
///
/// `layout.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-layout/layout.fbs")]
/// ```
pub mod layout;

#[cfg(feature = "ipc")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::unwrap_in_result)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::cast_possible_truncation)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[allow(missing_docs)]
#[rustfmt::skip]
#[path = "./generated/message.rs"]
/// A serialized sequence of arrays, each with its buffers.
///
/// `message.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-serde/message.fbs")]
/// ```
pub mod message;

pub use planus;
pub use planus::Builder as FlatBufferBuilder;
pub use planus::Offset as WIPOffset;
pub use planus::ReadAsRoot;
pub use planus::UnionOffset as UnionWIPOffset;
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

/// Reads a FlatBuffer root value from bytes.
pub fn root<'buf, T>(bytes: &'buf [u8]) -> Result<T, planus::Error>
where
    T: ReadAsRoot<'buf>,
{
    T::read_as_root(bytes)
}

/// Trait for reading a type from a FlatBuffer.
pub trait ReadFlatBuffer: Sized {
    /// The FlatBuffer type that this type can be read from.
    type Source<'a>: ReadAsRoot<'a>;
    /// The error type returned when reading fails.
    type Error: From<planus::Error>;

    /// Reads this type from a FlatBuffer source.
    fn read_flatbuffer<'buf>(fb: &Self::Source<'buf>) -> Result<Self, Self::Error>;

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
    type Target;

    /// Writes this type to a FlatBuffer builder.
    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>>;
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
        let bytes = fbb.finish(root_offset, None);
        Ok(FlatBuffer::align_from(ByteBuffer::from(bytes.to_vec())))
    }
}
