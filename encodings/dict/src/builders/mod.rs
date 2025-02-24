use bytes::BytesDictBuilder;
use primitive::PrimitiveDictBuilder;
use vortex_array::arrays::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_dtype::match_each_native_ptype;
use vortex_error::{vortex_bail, VortexResult};

use crate::DictArray;

mod bytes;
mod primitive;

pub trait DictEncoder {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef>;

    fn values(&mut self) -> VortexResult<ArrayRef>;
}

pub fn dict_encode(array: &dyn Array) -> VortexResult<DictArray> {
    let dict_builder: &mut dyn DictEncoder = if let Some(pa) = array.as_opt::<PrimitiveArray>() {
        match_each_native_ptype!(pa.ptype(), |$P| {
            &mut PrimitiveDictBuilder::<$P>::new(pa.dtype().nullability())
        })
    } else if let Some(vbv) = array.as_opt::<VarBinViewArray>() {
        &mut BytesDictBuilder::new(vbv.dtype().clone())
    } else if let Some(vb) = array.as_opt::<VarBinArray>() {
        &mut BytesDictBuilder::new(vb.dtype().clone())
    } else {
        vortex_bail!("Can only encode primitive or varbin/view arrays")
    };
    let codes = dict_builder.encode(array)?;
    DictArray::try_new(codes, dict_builder.values()?)
}
