use core::fmt;
use std::fmt::{Display, Formatter};

use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef, Canonical, IntoArray};

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
}

pub trait CompareFn<A> {
    /// Compares two arrays and returns a new boolean array with the result of the comparison.
    /// Or, returns None if comparison is not supported for these arrays.
    fn compare(
        &self,
        lhs: A,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> CompareFn<&dyn Array> for E
where
    E: for<'a> CompareFn<&'a E::Array>,
{
    fn compare(
        &self,
        lhs: &dyn Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = lhs
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");

        CompareFn::compare(self, array_ref, rhs, operator)
    }
}

pub fn compare(left: &dyn Array, right: &dyn Array, operator: Operator) -> VortexResult<ArrayRef> {
    if left.len() != right.len() {
        vortex_bail!("Compare operations only support arrays of the same length");
    }
    if !left.dtype().eq_ignore_nullability(right.dtype()) {
        vortex_bail!(
            "Compare operations only support arrays of the same type: {} != {}",
            left.dtype(),
            right.dtype()
        );
    }

    if left.dtype().is_struct() {
        vortex_bail!(
            "Compare does not support arrays with Struct DType, got: {} and {}",
            left.dtype(),
            right.dtype()
        )
    }

    let result_dtype =
        DType::Bool((left.dtype().is_nullable() || right.dtype().is_nullable()).into());

    if left.is_empty() {
        return Ok(Canonical::empty(&result_dtype).into_array());
    }

    let left_constant_null = left.as_constant().map(|l| l.is_null()).unwrap_or(false);
    let right_constant_null = right.as_constant().map(|r| r.is_null()).unwrap_or(false);
    if left_constant_null || right_constant_null {
        return Ok(ConstantArray::new(Scalar::null(result_dtype), left.len()).into_array());
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

/// Helper function to compare empty values with arrays that have external value length information
/// like `VarBin`.
pub fn compare_lengths_to_empty<P, I>(lengths: I, op: Operator) -> BooleanBuffer
where
    P: NativePType,
    I: Iterator<Item = P>,
{
    // All comparison can be expressed in terms of equality. "" is the absolute min of possible value.
    let cmp_fn = match op {
        Operator::Eq | Operator::Lte => |v| v == P::zero(),
        Operator::NotEq | Operator::Gt => |v| v != P::zero(),
        Operator::Gte => |_| true,
        Operator::Lt => |_| false,
    };

    lengths.map(cmp_fn).collect::<BooleanBuffer>()
}

/// Implementation of `CompareFn` using the Arrow crate.
fn arrow_compare(
    left: &dyn Array,
    right: &dyn Array,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();
    let lhs = Datum::try_new(left.to_array())?;
    let rhs = Datum::try_new(right.to_array())?;

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
fn check_compare_result(result: &dyn Array, lhs: &dyn Array, rhs: &dyn Array) {
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
    use crate::arrays::{BoolArray, ConstantArray};
    use crate::validity::Validity;
    use crate::ToCanonical;

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
        let arr = BoolArray::new(
            BooleanBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(&arr, &arr, Operator::Eq)
            .unwrap()
            .to_bool()
            .unwrap();

        assert_eq!(to_int_indices(matches), [1u64, 2, 3, 4]);

        let matches = compare(&arr, &arr, Operator::NotEq)
            .unwrap()
            .to_bool()
            .unwrap();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches), empty);

        let other = BoolArray::new(
            BooleanBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(&arr, &other, Operator::Lte)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [2u64, 3, 4]);

        let matches = compare(&arr, &other, Operator::Lt)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [4u64]);

        let matches = compare(&other, &arr, Operator::Gte)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [2u64, 3, 4]);

        let matches = compare(&other, &arr, Operator::Gt)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let compare = compare(&left, &right, Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);

        let compare = arrow_compare(&left.into_array(), &right.into_array(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);
    }

    #[rstest::rstest]
    #[case(Operator::Eq, vec![false, false, false, true])]
    #[case(Operator::NotEq, vec![true, true, true, false])]
    #[case(Operator::Gt, vec![true, true, true, false])]
    #[case(Operator::Gte, vec![true, true, true, true])]
    #[case(Operator::Lt, vec![false, false, false, false])]
    #[case(Operator::Lte, vec![false, false, false, true])]
    fn test_cmp_to_empty(#[case] op: Operator, #[case] expected: Vec<bool>) {
        let lengths: Vec<i32> = vec![1, 5, 7, 0];

        let output = compare_lengths_to_empty(lengths.iter().copied(), op);
        assert_eq!(Vec::from_iter(output.iter()), expected);
    }
}
