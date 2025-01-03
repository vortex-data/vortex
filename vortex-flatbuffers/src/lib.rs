//! A contiguously serialized Vortex array.
//!
//! See [message] and [footer] for the flatbuffer specifications.
//!
//! See the `vortex-file` crate for non-contiguous serialization.

#[cfg(feature = "array")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
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
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[rustfmt::skip]
#[path = "./generated/dtype.rs"]
/// A serialized data type.
///
/// `dtype.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-dtype/dtype.fbs")]
/// ```
pub mod dtype;

#[cfg(feature = "scalar")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[rustfmt::skip]
#[path = "./generated/scalar.rs"]
/// A serialized scalar.
///
/// `scalar.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-scalar/scalar.fbs")]
/// ```
pub mod scalar;

#[cfg(feature = "file")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[rustfmt::skip]
#[path = "./generated/footer.rs"]
/// A file format footer containining a serialized `vortex-serde` Layout.
///
/// `footer.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-serde/footer.fbs")]
/// ```
pub mod footer;

#[cfg(feature = "layout")]
#[allow(clippy::all)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[rustfmt::skip]
#[path = "./generated/layout.rs"]
/// A serialized sequence of arrays, each with its buffers.
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
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
#[rustfmt::skip]
#[path = "./generated/message.rs"]
/// A serialized sequence of arrays, each with its buffers.
///
/// `message.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-serde/message.fbs")]
/// ```
pub mod message;

use bytes::Bytes;
use flatbuffers::{root, FlatBufferBuilder, Follow, InvalidFlatbuffer, Verifiable, WIPOffset};

pub trait FlatBufferRoot {}

pub trait ReadFlatBuffer: Sized {
    type Source<'a>: Verifiable + Follow<'a>;
    type Error: From<InvalidFlatbuffer>;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error>;

    fn read_flatbuffer_bytes<'buf>(bytes: &'buf [u8]) -> Result<Self, Self::Error>
    where
        <Self as ReadFlatBuffer>::Source<'buf>: 'buf,
    {
        let fb = root::<Self::Source<'buf>>(bytes)?;
        Self::read_flatbuffer(&fb)
    }
}

pub trait WriteFlatBuffer {
    type Target<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>>;
}

pub trait WriteFlatBufferExt: WriteFlatBuffer + FlatBufferRoot {
    /// Write the flatbuffer into a [`Bytes`].
    fn write_flatbuffer_bytes(&self) -> Bytes;
}

impl<F: WriteFlatBuffer + FlatBufferRoot> WriteFlatBufferExt for F {
    fn write_flatbuffer_bytes(&self) -> Bytes {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = self.write_flatbuffer(&mut fbb);
        fbb.finish_minimal(root_offset);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        Bytes::from(vec).slice(start..end)
    }
}
