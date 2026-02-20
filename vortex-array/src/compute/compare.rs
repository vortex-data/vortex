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
    use vortex_buffer::buffer;

    use super::*;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::expr::Operator;
    use crate::scalar::Scalar;
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    #[test]
    fn test_bool_basic_comparisons() {
        let arr = BoolArray::new(
            BitBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = arr
            .to_array()
            .binary(arr.to_array(), Operator::Eq)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [1u64, 2, 3, 4]);

        let matches = arr
            .to_array()
            .binary(arr.to_array(), Operator::NotEq)
            .unwrap()
            .to_bool();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches).unwrap(), empty);

        let other = BoolArray::new(
            BitBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = arr
            .to_array()
            .binary(other.to_array(), Operator::Lte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = arr
            .to_array()
            .binary(other.to_array(), Operator::Lt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);

        let matches = other
            .to_array()
            .binary(arr.to_array(), Operator::Gte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = other
            .to_array()
            .binary(arr.to_array(), Operator::Gt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let result = left
            .to_array()
            .binary(right.to_array(), Operator::Gt)
            .unwrap();
        assert_eq!(result.len(), 10);
        let scalar = result.scalar_at(0).unwrap();
        assert_eq!(scalar.as_bool().value(), Some(false));
    }

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

    #[rstest]
    #[case(VarBinArray::from(vec!["a", "b"]).into_array(), VarBinViewArray::from_iter_str(["a", "b"]).into_array())]
    #[case(VarBinViewArray::from_iter_str(["a", "b"]).into_array(), VarBinArray::from(vec!["a", "b"]).into_array())]
    #[case(VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array())]
    #[case(VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array())]
    fn arrow_compare_different_encodings(#[case] left: ArrayRef, #[case] right: ArrayRef) {
        let res = left.binary(right, Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([true, true]);
        assert_arrays_eq!(res, expected);
    }

    #[ignore = "Arrow's ListView cannot be compared"]
    #[test]
    fn test_list_array_comparison() {
        let values1 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
        let offsets1 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
        let list1 = ListArray::try_new(
            values1.into_array(),
            offsets1.into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let values2 = PrimitiveArray::from_iter([1i32, 2, 3, 4, 7, 8]);
        let offsets2 = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
        let list2 = ListArray::try_new(
            values2.into_array(),
            offsets2.into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let result = list1
            .to_array()
            .binary(list2.to_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        let result = list1
            .to_array()
            .binary(list2.to_array(), Operator::NotEq)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);

        let result = list1
            .to_array()
            .binary(list2.to_array(), Operator::Lt)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);
    }

    #[ignore = "Arrow's ListView cannot be compared"]
    #[test]
    fn test_list_array_constant_comparison() {
        use std::sync::Arc;

        use crate::dtype::DType;
        use crate::dtype::PType;

        let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
        let offsets = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
        let list = ListArray::try_new(
            values.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![3i32.into(), 4i32.into()],
            Nullability::NonNullable,
        );
        let constant = ConstantArray::new(list_scalar, 3);

        let result = list
            .to_array()
            .binary(constant.to_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([false, true, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_struct_array_comparison() {
        let bool_field1 = BoolArray::from_iter([Some(true), Some(false), Some(true)]);
        let int_field1 = PrimitiveArray::from_iter([1i32, 2, 3]);

        let bool_field2 = BoolArray::from_iter([Some(true), Some(false), Some(false)]);
        let int_field2 = PrimitiveArray::from_iter([1i32, 2, 4]);

        let struct1 = StructArray::from_fields(&[
            ("bool_col", bool_field1.into_array()),
            ("int_col", int_field1.into_array()),
        ])
        .unwrap();

        let struct2 = StructArray::from_fields(&[
            ("bool_col", bool_field2.into_array()),
            ("int_col", int_field2.into_array()),
        ])
        .unwrap();

        let result = struct1
            .to_array()
            .binary(struct2.to_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        let result = struct1
            .to_array()
            .binary(struct2.to_array(), Operator::Gt)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_empty_struct_compare() {
        let empty1 = StructArray::try_new(
            FieldNames::from(Vec::<FieldName>::new()),
            Vec::new(),
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let empty2 = StructArray::try_new(
            FieldNames::from(Vec::<FieldName>::new()),
            Vec::new(),
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let result = empty1
            .to_array()
            .binary(empty2.to_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, true, true, true]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_empty_list() {
        let list = ListViewArray::new(
            BoolArray::from_iter(Vec::<bool>::new()).into_array(),
            buffer![0i32, 0i32, 0i32].into_array(),
            buffer![0i32, 0i32, 0i32].into_array(),
            Validity::AllValid,
        );

        let result = list
            .to_array()
            .binary(list.to_array(), Operator::Eq)
            .unwrap();
        assert!(result.scalar_at(0).unwrap().is_valid());
        assert!(result.scalar_at(1).unwrap().is_valid());
        assert!(result.scalar_at(2).unwrap().is_valid());
    }
}
