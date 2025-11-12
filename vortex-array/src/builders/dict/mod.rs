// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::bytes_dict_builder;
use primitive::primitive_dict_builder;
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use crate::arrays::{DictArray, PrimitiveVTable, VarBinVTable, VarBinViewVTable};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

mod bytes;
mod primitive;

#[derive(Clone)]
pub struct DictConstraints {
    pub max_bytes: usize,
    pub max_len: usize,
}

pub const UNCONSTRAINED: DictConstraints = DictConstraints {
    max_bytes: usize::MAX,
    max_len: usize::MAX,
};

pub trait DictEncoder: Send {
    /// Assign dictionary codes to the given input array.
    fn encode(&mut self, array: &dyn Array) -> ArrayRef;

    /// Clear the encoder state to make it ready for a new round of decoding.
    fn reset(&mut self) -> ArrayRef;
}

pub fn dict_encoder(array: &dyn Array, constraints: &DictConstraints) -> Box<dyn DictEncoder> {
    let dict_builder: Box<dyn DictEncoder> = if let Some(pa) = array.as_opt::<PrimitiveVTable>() {
        match_each_native_ptype!(pa.ptype(), |P| {
            primitive_dict_builder::<P>(pa.dtype().nullability(), constraints)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinViewVTable>() {
        bytes_dict_builder(vbv.dtype().clone(), constraints)
    } else if let Some(vb) = array.as_opt::<VarBinVTable>() {
        bytes_dict_builder(vb.dtype().clone(), constraints)
    } else {
        vortex_panic!("Can only encode primitive or varbin/view arrays")
    };
    dict_builder
}

pub fn dict_encode_with_constraints(
    array: &dyn Array,
    constraints: &DictConstraints,
) -> VortexResult<DictArray> {
    let mut encoder = dict_encoder(array, constraints);
    let codes = encoder.encode(array).to_primitive().narrow()?;
    // SAFETY: The encoding process will produce a value set of codes and values
    unsafe {
        Ok(DictArray::new_unchecked(
            codes.into_array(),
            encoder.reset(),
        ))
    }
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
