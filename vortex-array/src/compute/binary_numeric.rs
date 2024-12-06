use std::sync::Arc;

use arrow_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::{NumericOperator, Scalar};

use crate::array::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::encoding::{downcast_array, Encoding};
use crate::{ArrayDType, ArrayData, IntoArrayData as _, IntoCanonical};

pub trait BinaryNumericFn<Array> {
    fn binary_numeric(
        &self,
        array: &Array,
        other: &ArrayData,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayData>>;
}

impl<E: Encoding> BinaryNumericFn<ArrayData> for E
where
    E: BinaryNumericFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn binary_numeric(
        &self,
        lhs: &ArrayData,
        rhs: &ArrayData,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayData>> {
        let (array_ref, encoding) = downcast_array::<E>(lhs)?;
        BinaryNumericFn::binary_numeric(encoding, array_ref, rhs, op)
    }
}

/// Point-wise add two numeric arrays.
pub fn add(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
pub fn add_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs.cast(lhs.dtype())?, lhs.len()).into_array(),
        NumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
pub fn sub(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
pub fn sub_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs.cast(lhs.dtype())?, lhs.len()).into_array(),
        NumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
pub fn mul(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
pub fn mul_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs.cast(lhs.dtype())?, lhs.len()).into_array(),
        NumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
pub fn div(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
pub fn div_scalar(lhs: impl AsRef<ArrayData>, rhs: Scalar) -> VortexResult<ArrayData> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs.cast(lhs.dtype())?, lhs.len()).into_array(),
        NumericOperator::Mul,
    )
}

pub fn binary_numeric(
    lhs: &ArrayData,
    rhs: &ArrayData,
    op: NumericOperator,
) -> VortexResult<ArrayData> {
    if lhs.len() != rhs.len() {
        vortex_bail!("Numeric operations aren't supported on arrays of different lengths")
    }
    if !matches!(lhs.dtype(), DType::Primitive(_, _))
        || !matches!(rhs.dtype(), DType::Primitive(_, _))
        || lhs.dtype() != rhs.dtype()
    {
        vortex_bail!(
            "Numeric operations are only supported on two arrays sharing the same primitive-type: {} {}",
            lhs.dtype(),
            rhs.dtype()
        )
    }

    // Check if LHS supports the operation directly.
    if let Some(fun) = lhs.encoding().binary_numeric_fn() {
        if let Some(result) = fun.binary_numeric(lhs, rhs, op)? {
            return Ok(result);
        }
    }

    log::debug!(
        "No numeric implementation found for LHS {}, RHS {}, and operator {:?}",
        lhs.encoding().id(),
        rhs.encoding().id(),
        op,
    );

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_numeric(lhs.clone(), rhs.clone(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
fn arrow_numeric(
    lhs: ArrayData,
    rhs: ArrayData,
    operator: NumericOperator,
) -> VortexResult<ArrayData> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.into_canonical()?.into_arrow()?;
    let rhs = rhs.into_canonical()?.into_arrow()?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&lhs, &rhs)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&lhs, &rhs)?,
        NumericOperator::Div => arrow_arith::numeric::div(&lhs, &rhs)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&lhs, &rhs)?,
    };

    Ok(ArrayData::from_arrow(Arc::new(array) as ArrayRef, nullable))
}
