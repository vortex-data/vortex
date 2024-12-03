use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::arrow::FromArrowArray;
use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayVariant, IntoCanonical};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericOperator {
    Add,
    Sub,
    Div,
}

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
        let array_ref = <&E::Array>::try_from(lhs)?;
        let encoding = lhs
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        BinaryNumericFn::binary_numeric(encoding, array_ref, rhs, op)
    }
}

/// Point-wise add between two numeric arrays.
pub fn add(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Add)
}

/// Point-wise sub between two numeric arrays.
pub fn sub(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Sub)
}

/// Point-wise division between two numeric arrays.
pub fn div(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), NumericOperator::Div)
}

fn binary_numeric(lhs: &ArrayData, rhs: &ArrayData, op: NumericOperator) -> VortexResult<ArrayData> {
    if lhs.len() != rhs.len() {
        vortex_bail!("Numeric operations aren't supported on arrays of different lengths")
    }
    if !matches!(lhs.dtype(), DType::Primitive(_, _)) || !matches!(rhs.dtype(), DType::Primitive(_, _)) {
        vortex_bail!("Numeric operations are only supported on primitive arrays")
    }

    // Check if LHS supports the operation directly.
    if let Some(result) = lhs
        .encoding()
        .binary_numeric_fn()
        .and_then(|f| f.binary_numeric(lhs, rhs, op).transpose())
    {
        return result;
    }

    log::debug!(
        "No numeric implementation found for LHS {}, RHS {}, and operator {:?}",
        rhs.encoding().id(),
        lhs.encoding().id(),
        op,
    );

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_numeric(lhs.clone(), rhs.clone(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
pub(crate) fn arrow_numeric(
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
    };

    Ok(ArrayData::from_arrow(Arc::new(array) as ArrayRef, nullable))
}
