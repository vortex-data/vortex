use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ALPArray, ALPFloat, match_each_alp_float_ptype};

impl ArrayOperationsImpl for ALPArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ALPArray::try_new(
            self.encoded().slice(start, stop)?,
            self.exponents(),
            self.patches()
                .map(|p| p.slice(start, stop))
                .transpose()?
                .flatten(),
        )?
        .into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        if !self.encoded().is_valid(index)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }

        if let Some(patches) = self.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return patch.cast(self.dtype());
            }
        }

        let encoded_val = self.encoded().scalar_at(index)?;

        Ok(match_each_alp_float_ptype!(self.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                self.exponents(),
            ), self.dtype().nullability())
        }))
    }
}
