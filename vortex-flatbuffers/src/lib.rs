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
pub mod footer;

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
#[path = "./generated/message.rs"]
pub mod message;

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

pub trait FlatBufferToBytes {
    fn with_flatbuffer_bytes<R, Fn: FnOnce(&[u8]) -> R>(&self, f: Fn) -> R;
}

impl<F: WriteFlatBuffer + FlatBufferRoot> FlatBufferToBytes for F {
    fn with_flatbuffer_bytes<R, Fn: FnOnce(&[u8]) -> R>(&self, f: Fn) -> R {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = self.write_flatbuffer(&mut fbb);
        fbb.finish_minimal(root_offset);
        f(fbb.finished_data())
    }
}
