use bytes::bytes_dict_builder;
use primitive::primitive_dict_builder;
use vortex_array::arrays::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::compress::downscale_integer_array;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexResult, vortex_bail};

use crate::DictArray;

mod bytes;
mod primitive;

pub struct DictConstraints {
    pub max_bytes: usize,
    pub max_len: usize,
}

pub const UNCONSTRAINED: DictConstraints = DictConstraints {
    max_bytes: usize::MAX,
    max_len: usize::MAX,
};

pub trait DictEncoder: Send {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef>;

    fn values(&mut self) -> VortexResult<ArrayRef>;
}

pub fn dict_encoder(
    array: &dyn Array,
    constraints: &DictConstraints,
) -> VortexResult<Box<dyn DictEncoder>> {
    let dict_builder: Box<dyn DictEncoder> = if let Some(pa) = array.as_opt::<PrimitiveArray>() {
        match_each_native_ptype!(pa.ptype(), |$P| {
            primitive_dict_builder::<$P>(pa.dtype().nullability(), &constraints)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinViewArray>() {
        bytes_dict_builder(vbv.dtype().clone(), constraints)
    } else if let Some(vb) = array.as_opt::<VarBinArray>() {
        bytes_dict_builder(vb.dtype().clone(), constraints)
    } else {
        vortex_bail!("Can only encode primitive or varbin/view arrays")
    };
    Ok(dict_builder)
}

pub fn dict_encode_with_constraints(
    array: &dyn Array,
    constraints: &DictConstraints,
) -> VortexResult<DictArray> {
    let mut encoder = dict_encoder(array, constraints)?;
    let codes = downscale_integer_array(encoder.encode(array)?)?;
    DictArray::try_new(codes, encoder.values()?)
}

pub fn dict_encode(array: &dyn Array) -> VortexResult<DictArray> {
    let dict_array = dict_encode_with_constraints(array, &UNCONSTRAINED)?;
    if dict_array.len() != array.len() {
        vortex_bail!(
            "must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict_array.len(),
        );
    }
    Ok(dict_array)
}
