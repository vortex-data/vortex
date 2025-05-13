use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_scalar::Scalar;

use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, UnaryArgs};
use crate::stats::{Precision, Stat, StatsProvider};
use crate::vtable::VTable;
use crate::{Array, ArrayExt};

/// Sum an array.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be zero.
pub fn sum(array: &dyn Array) -> VortexResult<Scalar> {
    SUM_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &(),
        })?
        .unwrap_scalar()
}

struct Sum;

impl ComputeFnVTable for Sum {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        // Compute the expected dtype of the sum.
        let sum_dtype = self.return_dtype(args)?;

        // Short-circuit using array statistics.
        if let Some(Precision::Exact(sum)) = array.statistics().get(Stat::Sum) {
            return Ok(Scalar::new(sum_dtype, sum).into());
        }

        let sum_scalar = sum_impl(array, sum_dtype, kernels)?;

        // Update the statistics with the computed sum.
        array
            .statistics()
            .set(Stat::Sum, Precision::Exact(sum_scalar.value().clone()));

        Ok(sum_scalar.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;
        Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype: {}", array.dtype()))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        // The sum function always returns a single scalar value.
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

pub static SUM_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("sum".into(), ArcRef::new_ref(&Sum));
    for kernel in inventory::iter::<SumKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct SumKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(SumKernelRef);

pub trait SumKernel: VTable {
    /// # Preconditions
    ///
    /// * The array's DType is summable
    /// * The array is not all-null
    fn sum(&self, array: &Self::Array) -> VortexResult<Scalar>;
}

#[derive(Debug)]
pub struct SumKernelAdapter<V: VTable>(pub V);

impl<V: VTable + SumKernel> SumKernelAdapter<V> {
    pub const fn lift(&'static self) -> SumKernelRef {
        SumKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + SumKernel> Kernel for SumKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;
        let Some(array) = array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(V::sum(&self.0, array)?.into()))
    }
}

/// Sum an array.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be zero.
pub fn sum_impl(
    array: &dyn Array,
    sum_dtype: DType,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<Scalar> {
    if array.is_empty() {
        return if sum_dtype.is_float() {
            Ok(Scalar::new(sum_dtype, 0.0.into()))
        } else {
            Ok(Scalar::new(sum_dtype, 0.into()))
        };
    }

    // If the array is constant, we can compute the sum directly.
    if let Some(mut constant) = array.as_constant() {
        if constant.is_null() {
            // An all-null constant array has a sum of 0.
            return if sum_dtype.is_float() {
                Ok(Scalar::new(sum_dtype, 0.0.into()))
            } else {
                Ok(Scalar::new(sum_dtype, 0.into()))
            };
        }

        // TODO(ngates): I think we should delegate these to kernels, rather than hard-code.

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

    // Try to find a sum kernel
    let args = InvocationArgs {
        inputs: &[array.into()],
        options: &(),
    };
    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return output.unwrap_scalar();
        }
    }
    if let Some(output) = array.invoke(&SUM_FN, &args)? {
        return output.unwrap_scalar();
    }

    // Otherwise, canonicalize and try again.
    log::debug!("No sum implementation found for {}", array.encoding_id());
    if array.is_canonical() {
        // Panic to avoid recursion, but it should never be hit.
        vortex_panic!(
            "No sum implementation found for canonical array: {}",
            array.encoding_id()
        );
    }
    sum(array.to_canonical()?.as_ref())
}

#[cfg(test)]
mod test {
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::sum;

    #[test]
    fn sum_all_invalid() {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(0));
    }

    #[test]
    fn sum_all_invalid_float() {
        let array = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>().unwrap(), Some(0.0));
    }

    #[test]
    fn sum_constant() {
        let array = PrimitiveArray::from_iter([1, 1, 1, 1]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(4));
    }

    #[test]
    fn sum_constant_float() {
        let array = PrimitiveArray::from_iter([1., 1., 1., 1.]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>().unwrap(), Some(4.));
    }

    #[test]
    fn sum_boolean() {
        let array = BoolArray::from_iter([true, false, false, true]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>().unwrap(), Some(2));
    }
}
