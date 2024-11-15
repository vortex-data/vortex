use core::fmt;
use std::fmt::{Display, Formatter};

use arrow_ord::cmp;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::array::Constant;
use crate::arrow::FromArrowArray;
use crate::{ArrayDType, ArrayData, ArrayDef, IntoCanonical};

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

pub trait CompareFn {
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<ArrayData>;
}

pub trait MaybeCompareFn {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>>;
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

    if left.is_encoding(Constant::ID) && !right.is_encoding(Constant::ID) {
        return compare(right, left, operator.swap());
    }

    if let Some(selection) = left.with_dyn(|lhs| lhs.compare(right, operator)) {
        return selection;
    }

    if let Some(selection) = right.with_dyn(|rhs| rhs.compare(left, operator.swap())) {
        return selection;
    }

    // Fallback to arrow on canonical types
    let lhs = left.clone().into_canonical()?.into_arrow()?;
    let rhs = right.clone().into_canonical()?.into_arrow()?;

    let array = match operator {
        Operator::Eq => cmp::eq(&lhs.as_ref(), &rhs.as_ref())?,
        Operator::NotEq => cmp::neq(&lhs.as_ref(), &rhs.as_ref())?,
        Operator::Gt => cmp::gt(&lhs.as_ref(), &rhs.as_ref())?,
        Operator::Gte => cmp::gt_eq(&lhs.as_ref(), &rhs.as_ref())?,
        Operator::Lt => cmp::lt(&lhs.as_ref(), &rhs.as_ref())?,
        Operator::Lte => cmp::lt_eq(&lhs.as_ref(), &rhs.as_ref())?,
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
    use itertools::Itertools;
    use vortex_scalar::ScalarValue;

    use super::*;
    use crate::array::{BoolArray, ConstantArray};
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoArrayVariant};

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
        let arr = BoolArray::from_vec(
            vec![true, true, false, true, false],
            Validity::Array(BoolArray::from(vec![false, true, true, true, true]).into_array()),
        )
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

        let other = BoolArray::from_vec(
            vec![false, false, false, true, true],
            Validity::Array(BoolArray::from(vec![false, true, true, true, true]).into_array()),
        )
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

        let res = ConstantArray::try_from(compare(left, right, Operator::Gt).unwrap()).unwrap();
        assert_eq!(res.scalar_value(), &ScalarValue::Bool(false));
        assert_eq!(res.len(), 10);
    }
}
