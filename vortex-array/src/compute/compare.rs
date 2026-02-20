// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use arrow_array::BooleanArray;
use arrow_buffer::NullBuffer;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::dtype::IntegerPType;
use crate::expr::CompareOperator;

/// Helper function to compare empty values with arrays that have external value length information
/// like `VarBin`.
pub fn compare_lengths_to_empty<P, I>(lengths: I, op: CompareOperator) -> BitBuffer
where
    P: IntegerPType,
    I: Iterator<Item = P>,
{
    // All comparison can be expressed in terms of equality. "" is the absolute min of possible value.
    let cmp_fn = match op {
        CompareOperator::Eq | CompareOperator::Lte => |v| v == P::zero(),
        CompareOperator::NotEq | CompareOperator::Gt => |v| v != P::zero(),
        CompareOperator::Gte => |_| true,
        CompareOperator::Lt => |_| false,
    };

    lengths.map(cmp_fn).collect()
}

/// Compare two Arrow arrays element-wise using [`make_comparator`].
///
/// This function is required for nested types (Struct, List, FixedSizeList) because Arrow's
/// vectorized comparison kernels ([`cmp::eq`], [`cmp::neq`], etc.) do not support them.
///
/// The vectorized kernels are faster but only work on primitive types, so for non-nested types,
/// prefer using the vectorized kernels directly for better performance.
pub(crate) fn compare_nested_arrow_arrays(
    lhs: &dyn arrow_array::Array,
    rhs: &dyn arrow_array::Array,
    operator: CompareOperator,
) -> VortexResult<BooleanArray> {
    let compare_arrays_at = make_comparator(lhs, rhs, SortOptions::default())?;

    let cmp_fn = match operator {
        CompareOperator::Eq => Ordering::is_eq,
        CompareOperator::NotEq => Ordering::is_ne,
        CompareOperator::Gt => Ordering::is_gt,
        CompareOperator::Gte => Ordering::is_ge,
        CompareOperator::Lt => Ordering::is_lt,
        CompareOperator::Lte => Ordering::is_le,
    };

    let values = (0..lhs.len())
        .map(|i| cmp_fn(compare_arrays_at(i, i)))
        .collect();
    let nulls = NullBuffer::union(lhs.nulls(), rhs.nulls());

    Ok(BooleanArray::new(values, nulls))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(CompareOperator::Eq, vec![false, false, false, true])]
    #[case(CompareOperator::NotEq, vec![true, true, true, false])]
    #[case(CompareOperator::Gt, vec![true, true, true, false])]
    #[case(CompareOperator::Gte, vec![true, true, true, true])]
    #[case(CompareOperator::Lt, vec![false, false, false, false])]
    #[case(CompareOperator::Lte, vec![false, false, false, true])]
    fn test_cmp_to_empty(#[case] op: CompareOperator, #[case] expected: Vec<bool>) {
        let lengths: Vec<i32> = vec![1, 5, 7, 0];

        let output = compare_lengths_to_empty(lengths.iter().copied(), op);
        assert_eq!(Vec::from_iter(output.iter()), expected);
    }
}
