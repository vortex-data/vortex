mod between;
mod compare;
mod nan_count;

use vortex_array::compute::{NaNCountFn, ScalarAtFn, TakeFn, scalar_at, take};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ALPArray, ALPEncoding, ALPFloat, match_each_alp_float_ptype};

impl ComputeVTable for ALPEncoding {
    fn nan_count_fn(&self) -> Option<&dyn NaNCountFn<&dyn Array>> {
        Some(self)
    }
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&ALPArray> for ALPEncoding {
    fn scalar_at(&self, array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if !array.encoded().is_valid(index)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return patch.cast(array.dtype());
            }
        }

        let encoded_val = scalar_at(array.encoded(), index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                array.exponents(),
            ), array.dtype().nullability())
        }))
    }
}

impl TakeFn<&ALPArray> for ALPEncoding {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|p| {
                p.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array())
    }
}
