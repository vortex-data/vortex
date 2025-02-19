use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::BinaryNumericOperator;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::BinaryNumericFn;
use crate::{Array, IntoArray as _};

impl BinaryNumericFn<ConstantArray> for ConstantEncoding {
    fn binary_numeric(
        &self,
        array: &ConstantArray,
        rhs: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>> {
        let Some(rhs) = rhs.as_constant() else {
            return Ok(None);
        };

        Ok(Some(
            ConstantArray::new(
                array
                    .scalar()
                    .as_primitive()
                    .checked_binary_numeric(rhs.as_primitive(), op)?
                    .ok_or_else(|| vortex_err!("numeric overflow"))?,
                array.len(),
            )
            .into_array(),
        ))
    }
}
