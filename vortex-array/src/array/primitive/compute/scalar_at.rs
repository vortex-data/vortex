use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::compute::unary::ScalarAtFn;
use crate::variants::PrimitiveArrayTrait;
use crate::ArrayDType;

impl ScalarAtFn for PrimitiveArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(self.scalar_at_unchecked(index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        match_each_native_ptype!(self.ptype(), |$T| {
            Scalar::primitive(self.maybe_null_slice::<$T>()[index], self.dtype().nullability())
        })
    }
}
