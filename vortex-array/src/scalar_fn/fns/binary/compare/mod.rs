// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison kernels.
//!
//! [`execute_compare`] dispatches on the logical [`DType`] of its operands and evaluates every
//! comparison directly over Vortex canonical arrays — bit buffers for booleans, lane kernels from
//! `vortex-compute` for primitives and decimals, binary views for strings/bytes, and a row-wise
//! comparator for nested types. There is no Arrow fallback.
//!
//! Floating point values compare with Vortex's total ordering (`NaN` is the largest value,
//! `-0.0 < +0.0`, and equality is bitwise), matching [`Scalar`] comparison semantics.

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ScalarFn;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

mod boolean;
mod bytes;
mod decimal;
mod nested;
pub(crate) use nested::build_comparator as build_row_comparator;
mod primitive;
#[cfg(test)]
mod tests;

/// Trait for encoding-specific comparison kernels that operate in encoded space.
///
/// Implementations can compare an encoded array against another array (typically a constant)
/// without first decompressing. The adaptor normalizes operand order so `array` is always
/// the left-hand side, swapping the operator when necessary.
pub trait CompareKernel: VTable {
    fn compare(
        lhs: ArrayView<'_, Self>,
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
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Non-comparison operators are normal kernel misses, so decline them without constructing
        // an error (which would capture a backtrace, or panic under `VORTEX_PANIC_ON_ERR`).
        let Some(cmp_op) = parent.options.to_compare_operator() else {
            return Ok(None);
        };

        // Get the ScalarFnArray to access children
        let Some(scalar_fn_array) = parent.as_opt::<ScalarFn>() else {
            return Ok(None);
        };
        // Normalize so `array` is always LHS, swapping the operator if needed
        // TODO(joe): should be go this here or in the Rule/Kernel
        let (cmp_op, other) = match child_idx {
            0 => (cmp_op, scalar_fn_array.get_child(1)),
            1 => (cmp_op.swap(), scalar_fn_array.get_child(0)),
            _ => return Ok(None),
        };

        validate_comparison_dtypes(array.dtype(), other.dtype())?;

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
/// Handles empty, constant-null, and constant-constant directly, otherwise dispatches to a
/// native per-dtype kernel.
pub(crate) fn execute_compare(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    validate_comparison_dtypes(lhs.dtype(), rhs.dtype())?;

    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    if lhs.len() != rhs.len() {
        vortex_bail!(
            "compare operator requires equal lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        );
    }

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

    compare_arrays(lhs, rhs, op, ctx)
}

/// Dispatch a comparison to the native kernel for the operands' logical dtype.
fn compare_arrays(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullability = Nullability::from(lhs.dtype().is_nullable() || rhs.dtype().is_nullable());

    if lhs.dtype().is_extension() {
        let lhs = extension_storage(lhs, ctx)?;
        let rhs = extension_storage(rhs, ctx)?;

        return compare_arrays(&lhs, &rhs, op, ctx);
    }

    validate_comparison_dtypes(lhs.dtype(), rhs.dtype())?;

    match lhs.dtype() {
        // Every value of the null dtype is null, so every comparison result is null.
        DType::Null => Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            lhs.len(),
        )
        .into_array()),
        DType::Bool(_) => boolean::compare_bool(lhs, rhs, op, nullability, ctx),
        DType::Primitive(..) => primitive::compare_primitive(lhs, rhs, op, nullability, ctx),
        DType::Decimal(..) => decimal::compare_decimal(lhs, rhs, op, nullability, ctx),
        DType::Utf8(_) | DType::Binary(_) => bytes::compare_bytes(lhs, rhs, op, nullability, ctx),
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) | DType::Map(..) => {
            nested::compare_nested(lhs, rhs, op, nullability, ctx)
        }
        DType::Union(..) | DType::Variant(_) | DType::Extension(_) => {
            vortex_bail!("compare is not supported for dtype {}", lhs.dtype())
        }
    }
}

/// Returns the storage of an extension array or constant.
///
/// The caller must first require matching extension dtypes, ignoring nullability.
fn extension_storage(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if let Some(constant) = array.as_opt::<Constant>() {
        return Ok(ConstantArray::new(
            constant.scalar().as_extension().to_storage_scalar(),
            constant.len(),
        )
        .into_array());
    }
    let ext = array.clone().execute::<ExtensionArray>(ctx)?;
    Ok(ext.storage_array().clone())
}

/// Compares scalars with matching dtypes, ignoring nullability.
///
/// Returns null if either operand is null, and an error if the operand dtypes differ.
pub fn scalar_cmp(lhs: &Scalar, rhs: &Scalar, operator: CompareOperator) -> VortexResult<Scalar> {
    validate_comparison_dtypes(lhs.dtype(), rhs.dtype())?;

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

/// Checks the shared dtype requirement before comparison fast paths or encoding kernels run.
pub(super) fn validate_comparison_dtypes(lhs: &DType, rhs: &DType) -> VortexResult<()> {
    vortex_ensure!(
        lhs.eq_ignore_nullability(rhs),
        "Comparison requires equal dtypes ignoring nullability, got {lhs} and {rhs}"
    );

    Ok(())
}

/// Returns the predicate `Ordering -> bool` for a comparison operator.
#[inline]
pub(super) fn ordering_predicate(op: CompareOperator) -> fn(Ordering) -> bool {
    match op {
        CompareOperator::Eq => Ordering::is_eq,
        CompareOperator::NotEq => Ordering::is_ne,
        CompareOperator::Gt => Ordering::is_gt,
        CompareOperator::Gte => Ordering::is_ge,
        CompareOperator::Lt => Ordering::is_lt,
        CompareOperator::Lte => Ordering::is_le,
    }
}

/// Combine operand validities into the validity of a comparison result with the given result
/// nullability.
pub(super) fn compare_validity(
    lhs: Validity,
    rhs: Validity,
    nullability: Nullability,
) -> VortexResult<Validity> {
    Ok(lhs.and(rhs)?.union_nullability(nullability))
}
