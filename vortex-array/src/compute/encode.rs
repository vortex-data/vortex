use vortex_error::VortexResult;

use crate::vtable::VTableRef;
use crate::{Array, ArrayExt, ArrayRef, Encoding};

pub trait EncodeFn<A> {
    fn encode(&self, input: &dyn Array) -> VortexResult<Option<ArrayRef>>;
    fn encode_like(&self, input: &dyn Array, like: &A) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> EncodeFn<&dyn Array> for E
where
    E: EncodeFn<E::Array>,
{
    fn encode(&self, input: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        EncodeFn::encode(self, input)
    }

    fn encode_like(&self, input: &dyn Array, like: &&dyn Array) -> VortexResult<Option<ArrayRef>> {
        let like = like.as_::<E::Array>();
        EncodeFn::encode_like(self, input, like)
    }
}

pub fn encode(input: &dyn Array, encoding_vtable: VTableRef) -> VortexResult<Option<ArrayRef>> {
    match encoding_vtable.encode_fn() {
        None => return Ok(None),
        Some(encode_fn) => encode_fn.encode(input),
    }
}

pub fn encode_like(input: &dyn Array, like: &dyn Array) -> VortexResult<Option<ArrayRef>> {
    match like.vtable().encode_fn() {
        None => return Ok(None),
        Some(encode_fn) => encode_fn.encode_like(input, &like),
    }
}
