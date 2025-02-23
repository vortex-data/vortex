use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::stats::{Precision, Stat};
use crate::Array;

pub trait SumFn<A> {
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
pub fn sum(array: &dyn Array) -> VortexResult<Scalar> {
    // Compute the expected dtype of the sum.
    let sum_dtype = Stat::Sum.dtype(array.dtype());

    // If the sum_dtype is DType::Null, then the sum is always null.
    // This occurs when the array's dtype does not support summing, e.g. strings.
    if matches!(sum_dtype, DType::Null) {
        return Ok(Scalar::null(DType::Null));
    }

    // Short-circuit using array statistics.
    // TODO(ngates): enable this once statistics do not compute themselves.
    // if let Some(sum) = array.statistics().compute_stat(Stat::Sum) {
    //     return Ok(Scalar::new(sum_dtype, sum));
    // }

    // If the array is constant, we can compute the sum directly.
    if let Some(mut constant) = array.as_constant() {
        if constant.is_null() {
            // An all-null constant array has a sum of null.
            return Ok(Scalar::null(sum_dtype));
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
