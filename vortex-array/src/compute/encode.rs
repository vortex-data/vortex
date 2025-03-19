use vortex_error::VortexResult;

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

pub fn encode(_input: &dyn Array) -> VortexResult<Option<ArrayRef>> {
    todo!()
}

pub fn encode_like(_input: &dyn Array, _like: &dyn Array) -> VortexResult<Option<ArrayRef>> {
    todo!()
}
