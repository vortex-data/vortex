use core::fmt;
use std::fmt::{Display, Formatter};

use arrow_ord::cmp;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::encoding::Encoding;
use crate::{ArrayData, Canonical, IntoArrayData};

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd)]
pub enum Operator {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match &self {
            Operator::Eq => "=",
            Operator::NotEq => "!=",
            Operator::Gt => ">",
            Operator::Gte => ">=",
            Operator::Lt => "<",
            Operator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl Operator {
    pub fn inverse(self) -> Self {
        match self {
            Operator::Eq => Operator::NotEq,
            Operator::NotEq => Operator::Eq,
            Operator::Gt => Operator::Lte,
            Operator::Gte => Operator::Lt,
            Operator::Lt => Operator::Gte,
            Operator::Lte => Operator::Gt,
        }
    }

    /// Change the sides of the operator, where changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Self {
        match self {
            Operator::Eq => Operator::Eq,
            Operator::NotEq => Operator::NotEq,
            Operator::Gt => Operator::Lt,
            Operator::Gte => Operator::Lte,
            Operator::Lt => Operator::Gt,
            Operator::Lte => Operator::Gte,
        }
    }

    pub fn to_fn<T: PartialEq + PartialOrd>(&self) -> fn(T, T) -> bool {
        match self {
            Operator::Eq => |l, r| l == r,
            Operator::NotEq => |l, r| l != r,
            Operator::Gt => |l, r| l > r,
            Operator::Gte => |l, r| l >= r,
            Operator::Lt => |l, r| l < r,
            Operator::Lte => |l, r| l <= r,
        }
    }
}

pub trait CompareFn<Array> {
    /// Compares two arrays and returns a new boolean array with the result of the comparison.
    /// Or, returns None if comparison is not supported for these arrays.
    fn compare(
        &self,
        lhs: &Array,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>>;
}

impl<E: Encoding> CompareFn<ArrayData> for E
where
    E: CompareFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn compare(
        &self,
        lhs: &ArrayData,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        let (lhs_ref, encoding) = lhs.try_downcast_ref::<E>()?;
        CompareFn::compare(encoding, lhs_ref, rhs, operator)
    }
}

pub fn compare(
    left: impl AsRef<ArrayData>,
    right: impl AsRef<ArrayData>,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let left = left.as_ref();
    let right = right.as_ref();

    if left.len() != right.len() {
        vortex_bail!("Compare operations only support arrays of the same length");
    }
    if !left.dtype().eq_ignore_nullability(right.dtype()) {
        vortex_bail!("Compare operations only support arrays of the same type");
    }

    if left.dtype().is_struct() {
        vortex_bail!(
            "Compare does not support arrays with Strcut DType, got: {} and {}",
            left.dtype(),
            right.dtype()
        )
    }

    let result_dtype =
        DType::Bool((left.dtype().is_nullable() || right.dtype().is_nullable()).into());

    if left.is_empty() {
        return Ok(Canonical::empty(&result_dtype).into_array());
    }

    // Always try to put constants on the right-hand side so encodings can optimise themselves.
    if left.is_constant() && !right.is_constant() {
        return compare(right, left, operator.swap());
    }

    if let Some(result) = left
        .vtable()
        .compare_fn()
        .and_then(|f| f.compare(left, right, operator).transpose())
        .transpose()?
    {
        check_compare_result(&result, left, right);
        return Ok(result);
    }

    if let Some(result) = right
        .vtable()
        .compare_fn()
        .and_then(|f| f.compare(right, left, operator.swap()).transpose())
        .transpose()?
    {
        check_compare_result(&result, left, right);
        return Ok(result);
    }

    // Only log missing compare implementation if there's possibly better one than arrow,
    // i.e. lhs isn't arrow or rhs isn't arrow or constant
    if !(left.is_arrow() && (right.is_arrow() || right.is_constant())) {
        log::debug!(
            "No compare implementation found for LHS {}, RHS {}, and operator {} (or inverse)",
            right.encoding(),
            left.encoding(),
            operator.swap(),
        );
    }

    // Fallback to arrow on canonical types
    let result = arrow_compare(left, right, operator)?;
    check_compare_result(&result, left, right);
    Ok(result)
}

/// Implementation of `CompareFn` using the Arrow crate.
fn arrow_compare(
    left: &ArrayData,
    right: &ArrayData,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();
    let lhs = Datum::try_new(left.clone())?;
    let rhs = Datum::try_new(right.clone())?;

    let array = match operator {
        Operator::Eq => cmp::eq(&lhs, &rhs)?,
        Operator::NotEq => cmp::neq(&lhs, &rhs)?,
        Operator::Gt => cmp::gt(&lhs, &rhs)?,
        Operator::Gte => cmp::gt_eq(&lhs, &rhs)?,
        Operator::Lt => cmp::lt(&lhs, &rhs)?,
        Operator::Lte => cmp::lt_eq(&lhs, &rhs)?,
    };
    from_arrow_array_with_len(&array, left.len(), nullable)
}

#[inline(always)]
fn check_compare_result(result: &ArrayData, lhs: &ArrayData, rhs: &ArrayData) {
    debug_assert_eq!(
        result.len(),
        lhs.len(),
        "CompareFn result length ({}) mismatch for left encoding {}, left len {}, right encoding {}, right len {}",
        result.len(),
        lhs.encoding(),
        lhs.len(),
        rhs.encoding(),
        rhs.len()
    );
    debug_assert_eq!(
        result.dtype(),
        &DType::Bool((lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()),
        "CompareFn result dtype ({}) mismatch for left encoding {}, right encoding {}",
        result.dtype(),
        lhs.encoding(),
        rhs.encoding(),
    );
}

pub fn scalar_cmp(lhs: &Scalar, rhs: &Scalar, operator: Operator) -> Scalar {
    if lhs.is_null() | rhs.is_null() {
        Scalar::null(DType::Bool(Nullability::Nullable))
    } else {
        let b = match operator {
            Operator::Eq => lhs == rhs,
            Operator::NotEq => lhs != rhs,
            Operator::Gt => lhs > rhs,
            Operator::Gte => lhs >= rhs,
            Operator::Lt => lhs < rhs,
            Operator::Lte => lhs <= rhs,
        };

        Scalar::bool(
            b,
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use itertools::Itertools;

    use super::*;
    use crate::array::{BoolArray, ConstantArray};
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

    fn to_int_indices(indices_bits: BoolArray) -> Vec<u64> {
        let buffer = indices_bits.boolean_buffer();
        let null_buffer = indices_bits
            .validity()
            .to_logical(indices_bits.len())
            .unwrap()
            .to_null_buffer();
        let is_valid = |idx: usize| match null_buffer.as_ref() {
            None => true,
            Some(buffer) => buffer.is_valid(idx),
        };
        let filtered = buffer
            .iter()
            .enumerate()
            .flat_map(|(idx, v)| (v && is_valid(idx)).then_some(idx as u64))
            .collect_vec();
        filtered
    }

    #[test]
    fn test_bool_basic_comparisons() {
        let arr = BoolArray::try_new(
            BooleanBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        )
        .unwrap()
        .into_array();

        let matches = compare(&arr, &arr, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        assert_eq!(to_int_indices(matches), [1u64, 2, 3, 4]);

        let matches = compare(&arr, &arr, Operator::NotEq)
            .unwrap()
            .into_bool()
            .unwrap();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches), empty);

        let other = BoolArray::try_new(
            BooleanBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        )
        .unwrap()
        .into_array();

        let matches = compare(&arr, &other, Operator::Lte)
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [2u64, 3, 4]);

        let matches = compare(&arr, &other, Operator::Lt)
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [4u64]);

        let matches = compare(&other, &arr, Operator::Gte)
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [2u64, 3, 4]);

        let matches = compare(&other, &arr, Operator::Gt)
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let compare = compare(left.clone(), right.clone(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);

        let compare = arrow_compare(&left.into_array(), &right.into_array(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);
    }
}
