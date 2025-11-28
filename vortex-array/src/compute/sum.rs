// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use num_traits::CheckedAdd;
use num_traits::CheckedSub;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_scalar::NumericOperator;
use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::expr::stats::Stat;
use crate::stats::Precision;
use crate::stats::StatsProvider;
use crate::vtable::VTable;

static SUM_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("sum".into(), ArcRef::new_ref(&Sum));
    for kernel in inventory::iter::<SumKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    SUM_FN.kernels().len()
}

/// Sum an array with an initial value.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be the accumulator.
/// The accumulator must have a dtype compatible with the sum result dtype.
pub(crate) fn sum_with_accumulator(
    array: &dyn Array,
    accumulator: &Scalar,
) -> VortexResult<Scalar> {
    SUM_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), accumulator.into()],
            options: &(),
        })?
        .unwrap_scalar()
}

/// Sum an array, starting from zero.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be zero.
pub fn sum(array: &dyn Array) -> VortexResult<Scalar> {
    let sum_dtype = Stat::Sum
        .dtype(array.dtype())
        .ok_or_else(|| vortex_err!("Sum not supported for dtype: {}", array.dtype()))?;
    let zero = Scalar::zero_value(sum_dtype);
    sum_with_accumulator(array, &zero)
}

/// For unary compute functions, it's useful to just have this short-cut.
pub struct SumArgs<'a> {
    pub array: &'a dyn Array,
    pub accumulator: &'a Scalar,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for SumArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let accumulator = value.inputs[1]
            .scalar()
            .ok_or_else(|| vortex_err!("Expected input 1 to be a scalar"))?;
        Ok(SumArgs { array, accumulator })
    }
}

struct Sum;

impl ComputeFnVTable for Sum {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let SumArgs { array, accumulator } = args.try_into()?;

        // Compute the expected dtype of the sum.
        let sum_dtype = self.return_dtype(args)?;

        vortex_ensure!(
            &sum_dtype == accumulator.dtype(),
            "sum_dtype {sum_dtype} must match accumulator dtype {}",
            accumulator.dtype()
        );

        // Short-circuit using array statistics.
        if let Some(Precision::Exact(sum)) = array.statistics().get(Stat::Sum) {
            // For floats only use stats if accumulator is zero. otherwise we might have numerical stability issues.
            match sum_dtype {
                DType::Primitive(p, _) => {
                    if p.is_float() && accumulator.is_zero() {
                        return Ok(sum.into());
                    } else if p.is_int() {
                        let sum_from_stat = accumulator
                            .as_primitive()
                            .checked_add(&sum.as_primitive())
                            .map(Scalar::from);
                        return Ok(sum_from_stat
                            .unwrap_or_else(|| Scalar::null(sum_dtype))
                            .into());
                    }
                }
                DType::Decimal(..) => {
                    let sum_from_stat = accumulator
                        .as_decimal()
                        .checked_binary_numeric(&sum.as_decimal(), NumericOperator::Add)
                        .map(Scalar::from);
                    return Ok(sum_from_stat
                        .unwrap_or_else(|| Scalar::null(sum_dtype))
                        .into());
                }
                _ => unreachable!("Sum will always be a decimal or a primitive dtype"),
            }
        }

        let sum_scalar = sum_impl(array, accumulator, kernels)?;

        // Update the statistics with the computed sum. Stored statistic shouldn't include the accumulator.
        match sum_dtype {
            DType::Primitive(p, _) => {
                if p.is_float() && accumulator.is_zero() {
                    array
                        .statistics()
                        .set(Stat::Sum, Precision::Exact(sum_scalar.value().clone()));
                } else if p.is_int()
                    && let Some(less_accumulator) = sum_scalar
                        .as_primitive()
                        .checked_sub(&accumulator.as_primitive())
                {
                    array.statistics().set(
                        Stat::Sum,
                        Precision::Exact(Scalar::from(less_accumulator).value().clone()),
                    );
                }
            }
            DType::Decimal(..) => {
                if let Some(less_accumulator) = sum_scalar
                    .as_decimal()
                    .checked_binary_numeric(&accumulator.as_decimal(), NumericOperator::Sub)
                {
                    array.statistics().set(
                        Stat::Sum,
                        Precision::Exact(Scalar::from(less_accumulator).value().clone()),
                    )
                }
            }
            _ => unreachable!("Sum will always be a decimal or a primitive dtype"),
        }

        Ok(sum_scalar.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let SumArgs { array, .. } = args.try_into()?;
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

pub struct SumKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(SumKernelRef);

pub trait SumKernel: VTable {
    /// # Preconditions
    ///
    /// * The array's DType is summable
    /// * The array is not all-null
    /// * The accumulator must have a dtype compatible with the sum result dtype
    fn sum(&self, array: &Self::Array, accumulator: &Scalar) -> VortexResult<Scalar>;
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
        let SumArgs { array, accumulator } = args.try_into()?;
        let Some(array) = array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(V::sum(&self.0, array, accumulator)?.into()))
    }
}

/// Sum an array.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be the accumulator.
pub fn sum_impl(
    array: &dyn Array,
    accumulator: &Scalar,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<Scalar> {
    if array.is_empty() || array.all_invalid() || accumulator.is_null() {
        return Ok(accumulator.clone());
    }

    // Try to find a sum kernel
    let args = InvocationArgs {
        inputs: &[array.into(), accumulator.into()],
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
    sum_with_accumulator(array.to_canonical().as_ref(), accumulator)
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexUnwrap;
    use vortex_scalar::Scalar;

    use crate::IntoArray as _;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::sum;
    use crate::compute::sum_with_accumulator;

    #[test]
    fn sum_all_invalid() {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result, Scalar::primitive(0i64, Nullability::Nullable));
    }

    #[test]
    fn sum_all_invalid_float() {
        let array = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result, Scalar::primitive(0f64, Nullability::Nullable));
    }

    #[test]
    fn sum_constant() {
        let array = buffer![1, 1, 1, 1].into_array();
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
    }

    #[test]
    fn sum_constant_float() {
        let array = buffer![1., 1., 1., 1.].into_array();
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>(), Some(4.));
    }

    #[test]
    fn sum_boolean() {
        let array = BoolArray::from_iter([true, false, false, true]);
        let result = sum(array.as_ref()).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>(), Some(2));
    }

    #[test]
    fn sum_stats() {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter([1, 1, 1]).into_array(),
                PrimitiveArray::from_iter([2, 2, 2]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .vortex_unwrap();
        // compute sum with accumulator to populate stats
        sum_with_accumulator(
            array.as_ref(),
            &Scalar::primitive(2i64, Nullability::Nullable),
        )
        .unwrap();

        let sum_without_acc = sum(array.as_ref()).unwrap();
        assert_eq!(
            sum_without_acc,
            Scalar::primitive(9i64, Nullability::Nullable)
        );
    }
}
