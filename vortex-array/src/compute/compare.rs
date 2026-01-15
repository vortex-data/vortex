// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::BooleanArray;
use arrow_buffer::NullBuffer;
use arrow_ord::cmp;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrow::Datum;
use crate::arrow::IntoArrowArray;
use crate::arrow::from_arrow_array_with_len;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Options;
use crate::compute::Output;
use crate::vtable::VTable;

static COMPARE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("compare".into(), ArcRef::new_ref(&Compare));
    for kernel in inventory::iter::<CompareKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    COMPARE_FN.kernels().len()
}

/// Compares two arrays and returns a new boolean array with the result of the comparison.
/// Or, returns None if comparison is not supported for these arrays.
pub fn compare(left: &dyn Array, right: &dyn Array, operator: Operator) -> VortexResult<ArrayRef> {
    COMPARE_FN
        .invoke(&InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &operator,
        })?
        .unwrap_array()
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

pub struct CompareKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(CompareKernelRef);

pub trait CompareKernel: VTable {
    fn compare(
        &self,
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct CompareKernelAdapter<V: VTable>(pub V);

impl<V: VTable + CompareKernel> CompareKernelAdapter<V> {
    pub const fn lift(&'static self) -> CompareKernelRef {
        CompareKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + CompareKernel> Kernel for CompareKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = CompareArgs::try_from(args)?;
        let Some(array) = inputs.lhs.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::compare(&self.0, array, inputs.rhs, inputs.operator)?.map(|array| array.into()))
    }
}

struct Compare;

impl ComputeFnVTable for Compare {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let CompareArgs { lhs, rhs, operator } = CompareArgs::try_from(args)?;

        let return_dtype = self.return_dtype(args)?;

        if lhs.is_empty() {
            return Ok(Canonical::empty(&return_dtype).into_array().into());
        }

        let left_constant_null = lhs.as_constant().map(|l| l.is_null()).unwrap_or(false);
        let right_constant_null = rhs.as_constant().map(|r| r.is_null()).unwrap_or(false);
        if left_constant_null || right_constant_null {
            return Ok(ConstantArray::new(Scalar::null(return_dtype), lhs.len())
                .into_array()
                .into());
        }

        let right_is_constant = rhs.is_constant();

        // Always try to put constants on the right-hand side so encodings can optimise themselves.
        if lhs.is_constant() && !right_is_constant {
            return Ok(compare(rhs, lhs, operator.swap())?.into());
        }

        // First try lhs op rhs, then invert and try again.
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = lhs.invoke(&COMPARE_FN, args)? {
            return Ok(output);
        }

        // Try inverting the operator and swapping the arguments
        let inverted_args = InvocationArgs {
            inputs: &[rhs.into(), lhs.into()],
            options: &operator.swap(),
        };
        for kernel in kernels {
            if let Some(output) = kernel.invoke(&inverted_args)? {
                return Ok(output);
            }
        }
        if let Some(output) = rhs.invoke(&COMPARE_FN, &inverted_args)? {
            return Ok(output);
        }

        // Only log missing compare implementation if there's possibly better one than arrow,
        // i.e. lhs isn't arrow or rhs isn't arrow or constant
        if !(lhs.is_arrow() && (rhs.is_arrow() || right_is_constant)) {
            tracing::debug!(
                "No compare implementation found for LHS {}, RHS {}, and operator {} (or inverse)",
                lhs.encoding_id(),
                rhs.encoding_id(),
                operator,
            );
        }

        // Fallback to arrow on canonical types
        Ok(arrow_compare(lhs, rhs, operator)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let CompareArgs { lhs, rhs, .. } = CompareArgs::try_from(args)?;

        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            if lhs.dtype().is_float() && rhs.dtype().is_float() {
                vortex_bail!(
                    "Cannot compare different floating-point types ({}, {}). Consider using cast.",
                    lhs.dtype(),
                    rhs.dtype(),
                );
            }
            if lhs.dtype().is_int() && rhs.dtype().is_int() {
                vortex_bail!(
                    "Cannot compare different fixed-width types ({}, {}). Consider using cast.",
                    lhs.dtype(),
                    rhs.dtype()
                );
            }
            vortex_bail!(
                "Cannot compare different DTypes {} and {}",
                lhs.dtype(),
                rhs.dtype()
            );
        }

        Ok(DType::Bool(
            lhs.dtype().nullability() | rhs.dtype().nullability(),
        ))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let CompareArgs { lhs, rhs, .. } = CompareArgs::try_from(args)?;
        if lhs.len() != rhs.len() {
            vortex_bail!(
                "Compare operations only support arrays of the same length, got {} and {}",
                lhs.len(),
                rhs.len()
            );
        }
        Ok(lhs.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct CompareArgs<'a> {
    lhs: &'a dyn Array,
    rhs: &'a dyn Array,
    operator: Operator,
}

impl Options for Operator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<'a> TryFrom<&InvocationArgs<'a>> for CompareArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let lhs = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected first input to be an array"))?;
        let rhs = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected second input to be an array"))?;
        let operator = *value
            .options
            .as_any()
            .downcast_ref::<Operator>()
            .vortex_expect("Expected options to be an operator");

        Ok(CompareArgs { lhs, rhs, operator })
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

/// Implementation of `CompareFn` using the Arrow crate.
fn arrow_compare(
    left: &dyn Array,
    right: &dyn Array,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    assert_eq!(left.len(), right.len());

    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();

    // Arrow's vectorized comparison kernels (`cmp::eq`, etc.) are faster but don't support nested
    // types. For nested types, we fall back to `make_comparator` which does element-wise
    // comparison.
    let array = if left.dtype().is_nested() || right.dtype().is_nested() {
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
            Operator::Eq => cmp::eq(&lhs, &rhs)?,
            Operator::NotEq => cmp::neq(&lhs, &rhs)?,
            Operator::Gt => cmp::gt(&lhs, &rhs)?,
            Operator::Gte => cmp::gt_eq(&lhs, &rhs)?,
            Operator::Lt => cmp::lt(&lhs, &rhs)?,
            Operator::Lte => cmp::lt_eq(&lhs, &rhs)?,
        }
    };
    Ok(from_arrow_array_with_len(&array, left.len(), nullable))
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
    use crate::expr::get_item;
    use crate::expr::lt;
    use crate::expr::root;
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    #[test]
    fn test_bool_basic_comparisons() {
        let arr = BoolArray::from_bit_buffer(
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

        let other = BoolArray::from_bit_buffer(
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

        let compare = compare(left.as_ref(), right.as_ref(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);

        let compare = arrow_compare(&left.into_array(), &right.into_array(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);
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
        assert_eq!(res.to_bool().bit_buffer().true_count(), left.len());
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
        let bool_result = result.to_bool();
        assert!(bool_result.bit_buffer().value(0)); // [1,2] == [1,2]
        assert!(bool_result.bit_buffer().value(1)); // [3,4] == [3,4]
        assert!(!bool_result.bit_buffer().value(2)); // [5,6] != [7,8]

        // Test inequality
        let result = compare(list1.as_ref(), list2.as_ref(), Operator::NotEq).unwrap();
        let bool_result = result.to_bool();
        assert!(!bool_result.bit_buffer().value(0));
        assert!(!bool_result.bit_buffer().value(1));
        assert!(bool_result.bit_buffer().value(2));

        // Test less than
        let result = compare(list1.as_ref(), list2.as_ref(), Operator::Lt).unwrap();
        let bool_result = result.to_bool();
        assert!(!bool_result.bit_buffer().value(0)); // [1,2] < [1,2] = false
        assert!(!bool_result.bit_buffer().value(1)); // [3,4] < [3,4] = false
        assert!(bool_result.bit_buffer().value(2)); // [5,6] < [7,8] = true
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
        let bool_result = result.to_bool();
        assert!(!bool_result.bit_buffer().value(0)); // [1,2] != [3,4]
        assert!(bool_result.bit_buffer().value(1)); // [3,4] == [3,4]
        assert!(!bool_result.bit_buffer().value(2)); // [5,6] != [3,4]
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
        let bool_result = result.to_bool();
        assert!(bool_result.bit_buffer().value(0)); // {true, 1} == {true, 1}
        assert!(bool_result.bit_buffer().value(1)); // {false, 2} == {false, 2}
        assert!(!bool_result.bit_buffer().value(2)); // {true, 3} != {false, 4}

        // Test greater than
        let result = compare(struct1.as_ref(), struct2.as_ref(), Operator::Gt).unwrap();
        let bool_result = result.to_bool();
        assert!(!bool_result.bit_buffer().value(0)); // {true, 1} > {true, 1} = false
        assert!(!bool_result.bit_buffer().value(1)); // {false, 2} > {false, 2} = false
        assert!(bool_result.bit_buffer().value(2)); // {true, 3} > {false, 4} = true (bool field takes precedence)
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
        let result = result.to_bool();

        for idx in 0..5 {
            assert!(result.bit_buffer().value(idx));
        }
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
        assert!(result.scalar_at(0).is_valid());
        assert!(result.scalar_at(1).is_valid());
        assert!(result.scalar_at(2).is_valid());
    }

    #[test]
    fn test_different_floats_error_messages() {
        let result = compare(
            &buffer![0.0f32].into_array(),
            &buffer![0.0f64].into_array(),
            Operator::Lt,
        );
        assert!(result.as_ref().is_err_and(|err| {
            err.to_string()
                .contains("Cannot compare different floating-point types")
        }));

        let expr = lt(get_item("l", root()), get_item("r", root()));
        let result = expr.evaluate(
            &StructArray::from_fields(&[
                ("l", buffer![0.0f32].into_array()),
                ("r", buffer![0.0f64].into_array()),
            ])
            .unwrap()
            .into_array(),
        );
        assert!(result.as_ref().is_err_and(|err| {
            err.to_string()
                .contains("Cannot compare different floating-point types")
        }));
    }

    #[test]
    fn test_different_ints_error_messages() {
        let result = compare(
            &buffer![0u8].into_array(),
            &buffer![0u16].into_array(),
            Operator::Lt,
        );
        assert!(result.as_ref().is_err_and(|err| {
            err.to_string()
                .contains("Cannot compare different fixed-width types")
        }));

        let expr = lt(get_item("l", root()), get_item("r", root()));
        let result = expr.evaluate(
            &StructArray::from_fields(&[
                ("l", buffer![0u8].into_array()),
                ("r", buffer![0u16].into_array()),
            ])
            .unwrap()
            .into_array(),
        );
        assert!(result.as_ref().is_err_and(|err| {
            err.to_string()
                .contains("Cannot compare different fixed-width types")
        }));
    }
}
