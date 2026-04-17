// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::bytes_dict_builder;
use primitive::primitive_dict_builder;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::PType;
use crate::match_each_native_ptype;

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
    fn encode(&mut self, array: &ArrayRef) -> ArrayRef;

    /// Clear the encoder state to make it ready for a new round of decoding.
    fn reset(&mut self) -> ArrayRef;

    /// Returns the PType of the codes this encoder produces.
    fn codes_ptype(&self) -> PType;
}

pub fn dict_encoder(array: &ArrayRef, constraints: &DictConstraints) -> Box<dyn DictEncoder> {
    let dict_builder: Box<dyn DictEncoder> = if let Some(pa) = array.as_opt::<Primitive>() {
        match_each_native_ptype!(pa.ptype(), |P| {
            primitive_dict_builder::<P>(pa.dtype().nullability(), constraints)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinView>() {
        bytes_dict_builder(vbv.dtype().clone(), constraints)
    } else if let Some(vb) = array.as_opt::<VarBin>() {
        bytes_dict_builder(vb.dtype().clone(), constraints)
    } else {
        vortex_panic!("Can only encode primitive or varbin/view arrays")
    };
    dict_builder
}

/// Encode an array as a `DictArray` subject to the given constraints.
///
/// Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.
pub fn dict_encode_with_constraints(
    array: &ArrayRef,
    constraints: &DictConstraints,
) -> VortexResult<DictArray> {
    let mut encoder = dict_encoder(array, constraints);
    let encoded = encoder.encode(array);
    #[expect(deprecated)]
    let codes = encoded.to_primitive().narrow()?;
    // SAFETY: The encoding process will produce a value set of codes and values
    // All values in the dictionary are guaranteed to be referenced by at least one code
    // since we build the dictionary from the codes we observe during encoding
    unsafe {
        Ok(
            DictArray::new_unchecked(codes.into_array(), encoder.reset())
                .set_all_values_referenced(true),
        )
    }
}

pub fn dict_encode(array: &ArrayRef) -> VortexResult<DictArray> {
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
