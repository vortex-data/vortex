// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use arrow_array::BooleanArray;
use arrow_buffer::NullBuffer;
use arrow_ord::cmp;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrow::Datum;
use crate::arrow::IntoArrowArray;
use crate::arrow::from_arrow_array_with_len;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::vtable::VTable;

/// Trait for encoding-specific comparison kernels that operate in encoded space.
///
/// Implementations can compare an encoded array against another array (typically a constant)
/// without first decompressing. The adaptor normalizes operand order so `array` is always
/// the left-hand side, swapping the operator when necessary.
pub trait CompareKernel: VTable {
    fn compare(
        lhs: &Self::Array,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that bridges [`CompareKernel`] implementations to [`ExecuteParentKernel`].
///
/// When a `ScalarFnArray(Binary, cmp_op)` wraps a child that implements `CompareKernel`,
/// this adaptor extracts the comparison operator and other operand, normalizes operand order
/// (swapping the operator if the encoded array is on the RHS), and delegates to the kernel.
#[derive(Default, Debug)]
pub struct CompareExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for CompareExecuteAdaptor<V>
where
    V: CompareKernel,
{
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle comparison operators
        let Ok(cmp_op) = CompareOperator::try_from(*parent.options) else {
            return Ok(None);
        };

        // Get the ScalarFnArray to access children
        let Some(scalar_fn_array) = parent.as_opt::<ScalarFnVTable>() else {
            return Ok(None);
        };
        // Normalize so `array` is always LHS, swapping the operator if needed
        // TODO(joe): should be go this here or in the Rule/Kernel
        let (cmp_op, other) = match child_idx {
            0 => (cmp_op, scalar_fn_array.get_child(1)),
            1 => (cmp_op.swap(), scalar_fn_array.get_child(0)),
            _ => return Ok(None),
        };

        let len = array.len();
        let nullable = array.dtype().is_nullable() || other.dtype().is_nullable();

        // Empty array → empty bool result
        if len == 0 {
            return Ok(Some(
                Canonical::empty(&DType::Bool(nullable.into())).into_array(),
            ));
        }

        // Null constant on either side → all-null bool result
        if other.as_constant().is_some_and(|s| s.is_null()) {
            return Ok(Some(
                ConstantArray::new(Scalar::null(DType::Bool(nullable.into())), len).into_array(),
            ));
        }

        V::compare(array, other, cmp_op, ctx)
    }
}

/// Execute a compare operation between two arrays.
///
/// This is the entry point for compare operations from the binary expression.
/// Handles empty, constant-null, and constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_compare(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    if lhs.is_empty() {
        return Ok(Canonical::empty(&DType::Bool(nullable.into())).into_array());
    }

    let left_constant_null = lhs.as_constant().map(|l| l.is_null()).unwrap_or(false);
    let right_constant_null = rhs.as_constant().map(|r| r.is_null()).unwrap_or(false);
    if left_constant_null || right_constant_null {
        return Ok(
            ConstantArray::new(Scalar::null(DType::Bool(nullable.into())), lhs.len()).into_array(),
        );
    }

    // Constant-constant fast path
    if let (Some(lhs_const), Some(rhs_const)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>())
    {
        let result = scalar_cmp(lhs_const.scalar(), rhs_const.scalar(), op)?;
        return Ok(ConstantArray::new(result, lhs.len()).into_array());
    }

    arrow_compare_arrays(lhs, rhs, op)
}

/// Fall back to Arrow for comparison.
fn arrow_compare_arrays(
    left: &ArrayRef,
    right: &ArrayRef,
    operator: CompareOperator,
) -> VortexResult<ArrayRef> {
    assert_eq!(left.len(), right.len());

    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();

    // Arrow's vectorized comparison kernels don't support nested types.
    // For nested types, fall back to `make_comparator` which does element-wise comparison.
    let arrow_array: BooleanArray = if left.dtype().is_nested() || right.dtype().is_nested() {
        let rhs = right.to_array().into_arrow_preferred()?;
        let lhs = left.to_array().into_arrow(rhs.data_type())?;

        assert!(
            lhs.data_type().equals_datatype(rhs.data_type()),
            "lhs data_type: {}, rhs data_type: {}",
            lhs.data_type(),
            rhs.data_type()
        );

        compare_nested_arrow_arrays(lhs.as_ref(), rhs.as_ref(), operator)?
    } else {
        // Fast path: use vectorized kernels for primitive types.
        let lhs = Datum::try_new(left)?;
        let rhs = Datum::try_new_with_target_datatype(right, lhs.data_type())?;

        match operator {
            CompareOperator::Eq => cmp::eq(&lhs, &rhs)?,
            CompareOperator::NotEq => cmp::neq(&lhs, &rhs)?,
            CompareOperator::Gt => cmp::gt(&lhs, &rhs)?,
            CompareOperator::Gte => cmp::gt_eq(&lhs, &rhs)?,
            CompareOperator::Lt => cmp::lt(&lhs, &rhs)?,
            CompareOperator::Lte => cmp::lt_eq(&lhs, &rhs)?,
        }
    };

    from_arrow_array_with_len(&arrow_array, left.len(), nullable)
}

pub fn scalar_cmp(lhs: &Scalar, rhs: &Scalar, operator: CompareOperator) -> VortexResult<Scalar> {
    if lhs.is_null() | rhs.is_null() {
        return Ok(Scalar::null(DType::Bool(Nullability::Nullable)));
    }

    let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();

    // We use `partial_cmp` to ensure we do not lose a type mismatch error.
    let ordering = lhs.partial_cmp(rhs).ok_or_else(|| {
        vortex_err!(
            "Cannot compare scalars with incompatible types: {} and {}",
            lhs.dtype(),
            rhs.dtype()
        )
    })?;

    let b = match operator {
        CompareOperator::Eq => ordering.is_eq(),
        CompareOperator::NotEq => ordering.is_ne(),
        CompareOperator::Gt => ordering.is_gt(),
        CompareOperator::Gte => ordering.is_ge(),
        CompareOperator::Lt => ordering.is_lt(),
        CompareOperator::Lte => ordering.is_le(),
    };

    Ok(Scalar::bool(b, nullability))
}

/// Compare two Arrow arrays element-wise using [`make_comparator`].
///
/// This function is required for nested types (Struct, List, FixedSizeList) because Arrow's
/// vectorized comparison kernels ([`cmp::eq`], [`cmp::neq`], etc.) do not support them.
///
/// The vectorized kernels are faster but only work on primitive types, so for non-nested types,
/// prefer using the vectorized kernels directly for better performance.
pub fn compare_nested_arrow_arrays(
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
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::datetime::TimestampOptions;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::binary::compare::ConstantArray;
    use crate::scalar_fn::fns::binary::scalar_cmp;
    use crate::scalar_fn::fns::operators::CompareOperator;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    #[test]
    fn test_bool_basic_comparisons() {
        use vortex_buffer::BitBuffer;

        let arr = BoolArray::new(
            BitBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = arr
            .clone()
            .into_array()
            .binary(arr.clone().into_array(), Operator::Eq)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [1u64, 2, 3, 4]);

        let matches = arr
            .clone()
            .into_array()
            .binary(arr.clone().into_array(), Operator::NotEq)
            .unwrap()
            .to_bool();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches).unwrap(), empty);

        let other = BoolArray::new(
            BitBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = arr
            .clone()
            .into_array()
            .binary(other.clone().into_array(), Operator::Lte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = arr
            .clone()
            .into_array()
            .binary(other.clone().into_array(), Operator::Lt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);

        let matches = other
            .clone()
            .into_array()
            .binary(arr.clone().into_array(), Operator::Gte)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = other
            .into_array()
            .binary(arr.into_array(), Operator::Gt)
            .unwrap()
            .to_bool();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let result = left
            .into_array()
            .binary(right.into_array(), Operator::Gt)
            .unwrap();
        assert_eq!(result.len(), 10);
        let scalar = result.scalar_at(0).unwrap();
        assert_eq!(scalar.as_bool().value(), Some(false));
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
            .clone()
            .into_array()
            .binary(list2.clone().into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        let result = list1
            .clone()
            .into_array()
            .binary(list2.clone().into_array(), Operator::NotEq)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);

        let result = list1
            .into_array()
            .binary(list2.into_array(), Operator::Lt)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true]);
        assert_arrays_eq!(result, expected);
    }

    #[ignore = "Arrow's ListView cannot be compared"]
    #[test]
    fn test_list_array_constant_comparison() {
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
            .into_array()
            .binary(constant.into_array(), Operator::Eq)
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
            .clone()
            .into_array()
            .binary(struct2.clone().into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, false]);
        assert_arrays_eq!(result, expected);

        let result = struct1
            .into_array()
            .binary(struct2.into_array(), Operator::Gt)
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
            .into_array()
            .binary(empty2.into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, true, true, true]);
        assert_arrays_eq!(result, expected);
    }

    /// Regression test: `scalar_cmp` must error when comparing scalars with incompatible
    /// extension types (e.g., timestamps with different time units) rather than silently
    /// returning a wrong result.
    #[test]
    fn scalar_cmp_incompatible_extension_types_errors() {
        let ms_scalar = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            Scalar::from(1704067200000i64),
        );
        let s_scalar = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            Scalar::from(1704067200i64),
        );

        // Ordering comparisons must error on incompatible types.
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Gt).is_err());
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Lt).is_err());
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Gte).is_err());
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Lte).is_err());
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::Eq).is_err());
        assert!(scalar_cmp(&ms_scalar, &s_scalar, CompareOperator::NotEq).is_err());
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
            .clone()
            .into_array()
            .binary(list.into_array(), Operator::Eq)
            .unwrap();
        assert!(result.scalar_at(0).unwrap().is_valid());
        assert!(result.scalar_at(1).unwrap().is_valid());
        assert!(result.scalar_at(2).unwrap().is_valid());
    }
}
