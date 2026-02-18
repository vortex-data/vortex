// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;

use arrow_array::BooleanArray;
use arrow_buffer::NullBuffer;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Binary;
use crate::expr::ScalarFn;
use crate::expr::operators;
use crate::scalar::Scalar;

/// Compares two arrays and returns a new boolean array with the result of the comparison.
///
/// The returned array is lazy (a [`ScalarFnArray`]) and will be evaluated on demand.
pub fn compare(left: &dyn Array, right: &dyn Array, operator: Operator) -> VortexResult<ArrayRef> {
    let expr_op: operators::Operator = operator.into();
    Ok(ScalarFnArray::try_new(
        ScalarFn::new(Binary, expr_op),
        vec![left.to_array(), right.to_array()],
        left.len(),
    )?
    .into_array())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub enum Operator {
    /// Equality (`=`)
    Eq,
    /// Inequality (`!=`)
    NotEq,
    /// Greater than (`>`)
    Gt,
    /// Greater than or equal (`>=`)
    Gte,
    /// Less than (`<`)
    Lt,
    /// Less than or equal (`<=`)
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

/// Helper function to compare empty values with arrays that have external value length information
/// like `VarBin`.
pub fn compare_lengths_to_empty<P, I>(lengths: I, op: Operator) -> BitBuffer
where
    P: IntegerPType,
    I: Iterator<Item = P>,
{
    // All comparison can be expressed in terms of equality. "" is the absolute min of possible value.
    let cmp_fn = match op {
        Operator::Eq | Operator::Lte => |v| v == P::zero(),
        Operator::NotEq | Operator::Gt => |v| v != P::zero(),
        Operator::Gte => |_| true,
        Operator::Lt => |_| false,
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
    operator: Operator,
) -> VortexResult<BooleanArray> {
    let compare_arrays_at = make_comparator(lhs, rhs, SortOptions::default())?;

    let cmp_fn = match operator {
        Operator::Eq => Ordering::is_eq,
        Operator::NotEq => Ordering::is_ne,
        Operator::Gt => Ordering::is_gt,
        Operator::Gte => Ordering::is_ge,
        Operator::Lt => Ordering::is_lt,
        Operator::Lte => Ordering::is_le,
    };

    let values = (0..lhs.len())
        .map(|i| cmp_fn(compare_arrays_at(i, i)))
        .collect();
    let nulls = NullBuffer::union(lhs.nulls(), rhs.nulls());

    Ok(BooleanArray::new(values, nulls))
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

        Scalar::bool(b, lhs.dtype().nullability() | rhs.dtype().nullability())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::FieldName;
    use vortex_dtype::FieldNames;

    use super::*;
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
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    #[test]
    fn test_bool_basic_comparisons() {
        let arr = BoolArray::new(
            BitBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(arr.as_ref(), arr.as_ref(), Operator::Eq)
            .unwrap()
            .to_bool();

        assert_eq!(to_int_indices(matches).unwrap(), [1u64, 2, 3, 4]);

        let matches = compare(arr.as_ref(), arr.as_ref(), Operator::NotEq)
            .unwrap()
            .to_bool();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches).unwrap(), empty);

        let other = BoolArray::new(
            BitBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(arr.as_ref(), other.as_ref(), Operator::Lte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = compare(arr.as_ref(), other.as_ref(), Operator::Lt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);

        let matches = compare(other.as_ref(), arr.as_ref(), Operator::Gte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = compare(other.as_ref(), arr.as_ref(), Operator::Gt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let result = compare(left.as_ref(), right.as_ref(), Operator::Gt).unwrap();
        assert_eq!(result.len(), 10);
        let scalar = result.scalar_at(0).unwrap();
        assert_eq!(scalar.as_bool().value(), Some(false));
    }

    #[rstest]
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

    #[rstest]
    #[case(VarBinArray::from(vec!["a", "b"]).into_array(), VarBinViewArray::from_iter_str(["a", "b"]).into_array())]
    #[case(VarBinViewArray::from_iter_str(["a", "b"]).into_array(), VarBinArray::from(vec!["a", "b"]).into_array())]
    #[case(VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array())]
    #[case(VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array())]
    fn arrow_compare_different_encodings(#[case] left: ArrayRef, #[case] right: ArrayRef) {
        let res = compare(&left, &right, Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([true, true]);
        assert_arrays_eq!(res, expected);
    }

    #[ignore = "Arrow's ListView cannot be compared"]
    #[test]
    fn test_list_array_comparison() {
        // Create two simple list arrays with integers
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

        // Test equality - first two lists should be equal, third should be different
        let result = compare(list1.as_ref(), list2.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        // Test inequality
        let result = compare(list1.as_ref(), list2.as_ref(), Operator::NotEq).unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);

        // Test less than
        let result = compare(list1.as_ref(), list2.as_ref(), Operator::Lt).unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);
    }

    #[ignore = "Arrow's ListView cannot be compared"]
    #[test]
    fn test_list_array_constant_comparison() {
        use std::sync::Arc;

        use vortex_dtype::DType;
        use vortex_dtype::PType;

        // Create a list array
        let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
        let offsets = PrimitiveArray::from_iter([0i32, 2, 4, 6]);
        let list = ListArray::try_new(
            values.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        // Create a constant list scalar [3,4] that will be broadcasted
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![3i32.into(), 4i32.into()],
            Nullability::NonNullable,
        );
        let constant = ConstantArray::new(list_scalar, 3);

        // Compare list with constant - all should be compared to [3,4]
        let result = compare(list.as_ref(), constant.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([false, true, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_struct_array_comparison() {
        // Create two struct arrays with bool and int fields
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

        // Test equality
        let result = compare(struct1.as_ref(), struct2.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        // Test greater than
        let result = compare(struct1.as_ref(), struct2.as_ref(), Operator::Gt).unwrap();
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

        let result = compare(empty1.as_ref(), empty2.as_ref(), Operator::Eq).unwrap();
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

        // Compare two lists together
        let result = compare(list.as_ref(), list.as_ref(), Operator::Eq).unwrap();
        assert!(result.scalar_at(0).unwrap().is_valid());
        assert!(result.scalar_at(1).unwrap().is_valid());
        assert!(result.scalar_at(2).unwrap().is_valid());
    }
}
