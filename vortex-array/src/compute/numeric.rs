use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{NumericOperator, Scalar};

use crate::arrays::ConstantArray;
use crate::arrow::{Datum, from_arrow_array_with_len};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::vtable::VTable;
use crate::{Array, ArrayRef, IntoArray};

/// Point-wise add two numeric arrays.
pub fn add(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    numeric(lhs, rhs, NumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
pub fn add_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
pub fn sub(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    numeric(lhs, rhs, NumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
pub fn sub_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
pub fn mul(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    numeric(lhs, rhs, NumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
pub fn mul_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
pub fn div(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    numeric(lhs, rhs, NumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
pub fn div_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Mul,
    )
}

/// Point-wise numeric operation between two arrays of the same type and length.
pub fn numeric(lhs: &dyn Array, rhs: &dyn Array, op: NumericOperator) -> VortexResult<ArrayRef> {
    NUMERIC_FN
        .invoke(&InvocationArgs {
            inputs: &[lhs.into(), rhs.into()],
            options: &op,
        })?
        .unwrap_array()
}

pub struct NumericKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(NumericKernelRef);

pub trait NumericKernel: VTable {
    fn numeric(
        &self,
        array: &Self::Array,
        other: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct NumericKernelAdapter<V: VTable>(pub V);

impl<V: VTable + NumericKernel> NumericKernelAdapter<V> {
    pub const fn lift(&'static self) -> NumericKernelRef {
        NumericKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + NumericKernel> Kernel for NumericKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = NumericArgs::try_from(args)?;
        let Some(lhs) = inputs.lhs.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::numeric(&self.0, lhs, inputs.rhs, inputs.operator)?.map(|array| array.into()))
    }
}

pub static NUMERIC_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("numeric".into(), ArcRef::new_ref(&Numeric));
    for kernel in inventory::iter::<NumericKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Numeric;

impl ComputeFnVTable for Numeric {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let NumericArgs { lhs, rhs, operator } = NumericArgs::try_from(args)?;

        // Check if LHS supports the operation directly.
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = lhs.invoke(&NUMERIC_FN, args)? {
            return Ok(output);
        }

        // Check if RHS supports the operation directly.
        let inverted_args = InvocationArgs {
            inputs: &[rhs.into(), lhs.into()],
            options: &operator.swap(),
        };
        for kernel in kernels {
            if let Some(output) = kernel.invoke(&inverted_args)? {
                return Ok(output);
            }
        }
        if let Some(output) = rhs.invoke(&NUMERIC_FN, &inverted_args)? {
            return Ok(output);
        }

        log::debug!(
            "No numeric implementation found for LHS {}, RHS {}, and operator {:?}",
            lhs.encoding_id(),
            rhs.encoding_id(),
            operator,
        );

        // If neither side implements the trait, then we delegate to Arrow compute.
        Ok(arrow_numeric(lhs, rhs, operator)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let NumericArgs { lhs, rhs, .. } = NumericArgs::try_from(args)?;
        if !matches!(
            (lhs.dtype(), rhs.dtype()),
            (DType::Primitive(..), DType::Primitive(..)) | (DType::Decimal(..), DType::Decimal(..))
        ) || !lhs.dtype().eq_ignore_nullability(rhs.dtype())
        {
            vortex_bail!(
                "Numeric operations are only supported on two arrays sharing the same numeric type: {} {}",
                lhs.dtype(),
                rhs.dtype()
            )
        }
        Ok(lhs.dtype().union_nullability(rhs.dtype().nullability()))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let NumericArgs { lhs, rhs, .. } = NumericArgs::try_from(args)?;
        if lhs.len() != rhs.len() {
            vortex_bail!(
                "Numeric operations aren't supported on arrays of different lengths {} {}",
                lhs.len(),
                rhs.len()
            )
        }
        Ok(lhs.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct NumericArgs<'a> {
    lhs: &'a dyn Array,
    rhs: &'a dyn Array,
    operator: NumericOperator,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for NumericArgs<'a> {
    type Error = VortexError;

    fn try_from(args: &InvocationArgs<'a>) -> VortexResult<Self> {
        if args.inputs.len() != 2 {
            vortex_bail!("Numeric operations require exactly 2 inputs");
        }
        let lhs = args.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("LHS is not an array"))?;
        let rhs = args.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("RHS is not an array"))?;
        let operator = *args
            .options
            .as_any()
            .downcast_ref::<NumericOperator>()
            .ok_or_else(|| vortex_err!("Operator is not a numeric operator"))?;
        Ok(Self { lhs, rhs, operator })
    }
}

impl Options for NumericOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Implementation of `BinaryNumericFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
fn arrow_numeric(
    lhs: &dyn Array,
    rhs: &dyn Array,
    operator: NumericOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs)?;
    let right = Datum::try_new(rhs)?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        NumericOperator::RSub => arrow_arith::numeric::sub(&right, &left)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        NumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
        NumericOperator::RDiv => arrow_arith::numeric::div(&right, &left)?,
    };

    from_arrow_array_with_len(array, len, nullable)
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::sub_scalar;

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let results = sub_scalar(&values, 1u16.into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<u16>()
            .to_vec();
        assert_eq!(results, &[0u16, 1, 2]);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let results = sub_scalar(&values, (-1i64).into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<i64>()
            .to_vec();
        assert_eq!(results, &[2i64, 3, 4]);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(values.as_ref(), Some(1u16).into())
            .unwrap()
            .to_primitive()
            .unwrap();

        let actual = (0..result.len())
            .map(|index| result.scalar_at(index).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                Scalar::from(Some(0u16)),
                Scalar::from(Some(1u16)),
                Scalar::from(None::<u16>),
                Scalar::from(Some(2u16))
            ]
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let results = sub_scalar(&values, to_subtract.into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<f64>()
            .to_vec();
        assert_eq!(results, &[2.0f64, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = sub_scalar(&values, f32::MAX.into()).unwrap();
    }

    #[test]
    fn test_scalar_subtract_type_mismatch_fails() {
        let values = buffer![1u64, 2, 3].into_array();
        // Subtracting incompatible dtypes should fail
        let _results =
            sub_scalar(&values, 1.5f64.into()).expect_err("Expected type mismatch error");
    }
}
