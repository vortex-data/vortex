// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::BooleanArray;
use arrow_ord::cmp;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::ScalarFnVTable;
use crate::arrow::Datum;
use crate::arrow::IntoArrowArray;
use crate::arrow::from_arrow_array_with_len;
use crate::compute::Operator;
use crate::compute::compare_nested_arrow_arrays;
use crate::compute::scalar_cmp;
use crate::expr::Binary;
use crate::kernel::ExecuteParentKernel;
use crate::scalar::Scalar;
use crate::vtable::VTable;

/// Trait for encoding-specific comparison kernels that operate in encoded space.
///
/// Implementations can compare an encoded array against another array (typically a constant)
/// without first decompressing. The adaptor normalizes operand order so `array` is always
/// the left-hand side, swapping the operator when necessary.
pub trait CompareKernel: VTable {
    fn compare(
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
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
        let Some(cmp_op) = parent.options.maybe_cmp_operator() else {
            return Ok(None);
        };

        // Get the ScalarFnArray to access children
        let Some(scalar_fn_array) = parent.as_opt::<ScalarFnVTable>() else {
            return Ok(None);
        };
        let children = scalar_fn_array.children();

        // Normalize so `array` is always LHS, swapping the operator if needed
        // TODO(joe): should be go this here or in the Rule/Kernel
        let (cmp_op, other) = match child_idx {
            0 => (cmp_op, &children[1]),
            1 => (cmp_op.swap(), &children[0]),
            _ => return Ok(None),
        };

        let len = array.len();
        let nullable = array.dtype().is_nullable() || other.dtype().is_nullable();

        // Empty array → empty bool result
        if len == 0 {
            return Ok(Some(
                Canonical::empty(&vortex_dtype::DType::Bool(nullable.into())).into_array(),
            ));
        }

        // Null constant on either side → all-null bool result
        if other.as_constant().is_some_and(|s| s.is_null()) {
            return Ok(Some(
                ConstantArray::new(
                    Scalar::null(vortex_dtype::DType::Bool(nullable.into())),
                    len,
                )
                .into_array(),
            ));
        }

        V::compare(array, other.as_ref(), cmp_op, ctx)
    }
}

/// Execute a compare operation between two arrays.
///
/// This is the entry point for compare operations from the binary expression.
/// Handles empty, constant-null, and constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_compare(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: Operator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    if lhs.is_empty() {
        return Ok(Canonical::empty(&vortex_dtype::DType::Bool(nullable.into())).into_array());
    }

    let left_constant_null = lhs.as_constant().map(|l| l.is_null()).unwrap_or(false);
    let right_constant_null = rhs.as_constant().map(|r| r.is_null()).unwrap_or(false);
    if left_constant_null || right_constant_null {
        return Ok(ConstantArray::new(
            Scalar::null(vortex_dtype::DType::Bool(nullable.into())),
            lhs.len(),
        )
        .into_array());
    }

    // Constant-constant fast path
    if let (Some(lhs_const), Some(rhs_const)) = (
        lhs.as_opt::<ConstantVTable>(),
        rhs.as_opt::<ConstantVTable>(),
    ) {
        let result = scalar_cmp(lhs_const.scalar(), rhs_const.scalar(), op);
        return Ok(ConstantArray::new(result, lhs.len()).into_array());
    }

    arrow_compare_arrays(lhs, rhs, op)
}

/// Fall back to Arrow for comparison.
fn arrow_compare_arrays(
    left: &dyn Array,
    right: &dyn Array,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    assert_eq!(left.len(), right.len());

    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();

    // Arrow's vectorized comparison kernels don't support nested types.
    // For nested types, fall back to `make_comparator` which does element-wise comparison.
    let array: BooleanArray = if left.dtype().is_nested() || right.dtype().is_nested() {
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
    from_arrow_array_with_len(&array, left.len(), nullable)
}
