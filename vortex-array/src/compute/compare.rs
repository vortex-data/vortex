use core::fmt;
use std::fmt::{Display, Formatter};

use arrow_ord::cmp;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::arrow::{Datum, FromArrowArray};
use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData};

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
        let lhs_ref = <&E::Array>::try_from(lhs)?;
        let encoding = lhs
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
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

    // TODO(adamg): This is a placeholder until we figure out type coercion and casting
    if !left.dtype().eq_ignore_nullability(right.dtype()) {
        vortex_bail!("Compare operations only support arrays of the same type");
    }

    // Always try to put constants on the right-hand side so encodings can optimise themselves.
    if left.is_constant() && !right.is_constant() {
        return compare(right, left, operator.swap());
    }

    // If the RHS is constant and the LHS is Arrow, we can't do any better than arrow_compare.
    if left.is_arrow() && right.is_constant() {
        return arrow_compare(left, right, operator);
    }

    if let Some(result) = left
        .encoding()
        .compare_fn()
        .and_then(|f| f.compare(left, right, operator).transpose())
    {
        return result;
    } else {
        log::debug!(
            "No compare implementation found for LHS {}, RHS {}, and operator {}",
            left.encoding().id(),
            right.encoding().id(),
            operator,
        );
    }

    if let Some(result) = right
        .encoding()
        .compare_fn()
        .and_then(|f| f.compare(right, left, operator.swap()).transpose())
    {
        return result;
    } else {
        log::debug!(
            "No compare implementation found for LHS {}, RHS {}, and operator {}",
            right.encoding().id(),
            left.encoding().id(),
            operator.swap(),
        );
    }

    // Fallback to arrow on canonical types
    arrow_compare(left, right, operator)
}

/// Implementation of `CompareFn` using the Arrow crate.
pub(crate) fn arrow_compare(
    lhs: &ArrayData,
    rhs: &ArrayData,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let lhs = Datum::try_from(lhs.clone())?;
    let rhs = Datum::try_from(rhs.clone())?;

    let array = match operator {
        Operator::Eq => cmp::eq(&lhs, &rhs)?,
        Operator::NotEq => cmp::neq(&lhs, &rhs)?,
        Operator::Gt => cmp::gt(&lhs, &rhs)?,
        Operator::Gte => cmp::gt_eq(&lhs, &rhs)?,
        Operator::Lt => cmp::lt(&lhs, &rhs)?,
        Operator::Lte => cmp::lt_eq(&lhs, &rhs)?,
    };

    Ok(ArrayData::from_arrow(&array, true))
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

        Scalar::bool(b, Nullability::Nullable)
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use itertools::Itertools;

    use super::*;
    use crate::array::{BoolArray, ConstantArray};
    use crate::validity::Validity;
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

    fn to_int_indices(indices_bits: BoolArray) -> Vec<u64> {
        let buffer = indices_bits.boolean_buffer();
        let null_buffer = indices_bits
            .validity()
            .to_logical(indices_bits.len())
            .to_null_buffer()
            .unwrap();
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

        let compare = compare(left, right, Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);
    }
}
