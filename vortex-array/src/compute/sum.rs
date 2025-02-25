use vortex_dtype::PType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_scalar::Scalar;

use crate::Array;
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat};

pub trait SumFn<A> {
    /// # Preconditions
    ///
    /// * The array's DType is summable
    /// * The array is not all-null
    fn sum(&self, array: A) -> VortexResult<Scalar>;
}

impl<E: Encoding> SumFn<&dyn Array> for E
where
    E: for<'a> SumFn<&'a E::Array>,
{
    fn sum(&self, array: &dyn Array) -> VortexResult<Scalar> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        SumFn::sum(self, array_ref)
    }
}

/// Sum an array.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be zero.
pub fn sum(array: &dyn Array) -> VortexResult<Scalar> {
    // Compute the expected dtype of the sum.
    let sum_dtype = Stat::Sum
        .dtype(array.dtype())
        .ok_or_else(|| vortex_err!("Sum not supported for dtype: {}", array.dtype()))?;

    // Short-circuit using array statistics.
    if let Some(Precision::Exact(sum)) = array.statistics().get_stat(Stat::Sum) {
        return Ok(Scalar::new(sum_dtype, sum));
    }

    // If the array is constant, we can compute the sum directly.
    if let Some(mut constant) = array.as_constant() {
        if constant.is_null() {
            // An all-null constant array has a sum of 0.
            return if PType::try_from(&sum_dtype)
                .vortex_expect("must be primitive")
                .is_float()
            {
                Ok(Scalar::new(sum_dtype, 0.0.into()))
            } else {
                Ok(Scalar::new(sum_dtype, 0.into()))
            };
        }

        // If it's an extension array, then unwrap it into the storage scalar.
        if let Some(extension) = constant.as_extension_opt() {
            constant = extension.storage();
        }

        // If it's a boolean array, then the true count is the sum, which is the length.
        if let Some(bool) = constant.as_bool_opt() {
            return if bool.value().vortex_expect("already checked for null value") {
                // Constant true
                Ok(Scalar::new(sum_dtype, array.len().into()))
            } else {
                // Constant false
                Ok(Scalar::new(sum_dtype, 0.into()))
            };
        }

        // If it's a primitive array, then the sum is the constant value times the length.
        if let Some(primitive) = constant.as_primitive_opt() {
            match primitive.ptype() {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                    let value = primitive
                        .pvalue()
                        .vortex_expect("already checked for null value")
                        .as_u64()
                        .vortex_expect("Failed to cast constant value to u64");

                    // Overflow results in a null sum.
                    let sum = value.checked_mul(array.len() as u64);

                    return Ok(Scalar::new(sum_dtype, sum.into()));
                }
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                    let value = primitive
                        .pvalue()
                        .vortex_expect("already checked for null value")
                        .as_i64()
                        .vortex_expect("Failed to cast constant value to i64");

                    // Overflow results in a null sum.
                    let sum = value.checked_mul(array.len() as i64);

                    return Ok(Scalar::new(sum_dtype, sum.into()));
                }
                PType::F16 | PType::F32 | PType::F64 => {
                    let value = primitive
                        .pvalue()
                        .vortex_expect("already checked for null value")
                        .as_f64()
                        .vortex_expect("Failed to cast constant value to f64");

                    let sum = value * (array.len() as f64);

                    return Ok(Scalar::new(sum_dtype, sum.into()));
                }
            }
        }

        // For the unsupported types, we should have exited earlier.
        unreachable!("Unsupported sum constant: {}", constant.dtype());
    }

    // Try to use the sum function from the vtable.
    let sum = if let Some(f) = array.vtable().sum_fn() {
        f.sum(array)?
    } else {
        // Otherwise, canonicalize and try again.
        log::debug!("No sum implementation found for {}", array.encoding());

        let array = array.to_canonical()?;
        if let Some(f) = array.as_ref().vtable().sum_fn() {
            f.sum(array.as_ref())?
        } else {
            vortex_bail!(
                "No sum function for canonical array: {}",
                array.as_ref().encoding(),
            )
        }
    };

    if sum.dtype() != &sum_dtype {
        vortex_panic!(
            "Sum function of {} returned scalar with wrong dtype: {:?}",
            array.encoding(),
            sum.dtype()
        );
    }

    // Update the statistics with the computed sum.
    array
        .statistics()
        .set_stat(Stat::Sum, Precision::Exact(sum.value().clone()));

    Ok(sum)
}

#[cfg(test)]
mod test {
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::sum;

    #[test]
    fn sum_all_invalid() {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(0));
    }

    #[test]
    fn sum_all_invalid_float() {
        let array = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]);
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>().unwrap(), Some(0.0));
    }

    #[test]
    fn sum_constant() {
        let array = PrimitiveArray::from_iter([1, 1, 1, 1]);
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(4));
    }

    #[test]
    fn sum_constant_float() {
        let array = PrimitiveArray::from_iter([1., 1., 1., 1.]);
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>().unwrap(), Some(4.));
    }

    #[test]
    fn sum_boolean() {
        let array = BoolArray::from_iter([true, false, false, true]);
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(2));
    }
}
