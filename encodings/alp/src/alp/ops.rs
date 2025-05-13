use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ALPArray, ALPFloat, ALPVTable, match_each_alp_float_ptype};

impl OperationsVTable<ALPVTable> for ALPVTable {
    fn slice(array: &ALPArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ALPArray::try_new(
            array.encoded().slice(start, stop)?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(start, stop))
                .transpose()?
                .flatten(),
        )?
        .into_array())
    }

    fn scalar_at(array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if !array.encoded().is_valid(index)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return patch.cast(array.dtype());
            }
        }

        let encoded_val = array.encoded().scalar_at(index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                array.exponents(),
            ), array.dtype().nullability())
        }))
    }
}
