use bytes::BytesDictBuilder;
use primitive::PrimitiveDictBuilder;
use vortex_array::arrays::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexResult, vortex_bail};

use crate::DictArray;

mod bytes;
mod primitive;

pub trait DictEncoder {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef>;

    fn values(&mut self) -> VortexResult<ArrayRef>;
}

pub fn dict_encode_max_sized(array: &dyn Array, max_dict_bytes: usize) -> VortexResult<DictArray> {
    let dict_builder: &mut dyn DictEncoder = if let Some(pa) = array.as_opt::<PrimitiveArray>() {
        match_each_native_ptype!(pa.ptype(), |$P| {
            &mut PrimitiveDictBuilder::<$P>::new(pa.dtype().nullability(), max_dict_bytes)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinViewArray>() {
        &mut BytesDictBuilder::new(vbv.dtype().clone(), max_dict_bytes)
    } else if let Some(vb) = array.as_opt::<VarBinArray>() {
        &mut BytesDictBuilder::new(vb.dtype().clone(), max_dict_bytes)
    } else {
        vortex_bail!("Can only encode primitive or varbin/view arrays")
    };
    let codes = dict_builder.encode(array)?;
    DictArray::try_new(codes, dict_builder.values()?)
}

pub fn dict_encode(array: &dyn Array) -> VortexResult<DictArray> {
    let dict_array = dict_encode_max_sized(array, usize::MAX)?;
    if dict_array.len() != array.len() {
        vortex_bail!(
            "must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict_array.len(),
        );
    }
    Ok(dict_array)
}
