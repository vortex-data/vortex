// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison kernels.
//!
//! `execute_compare` dispatches on the logical [`DType`] of its operands and evaluates every
//! comparison directly over Vortex canonical arrays: bit buffers for booleans, [`RowFn`] for
//! primitives, lane kernels from `vortex-compute` for decimals, binary views for strings/bytes, and
//! a row-wise comparator for nested types. There is no Arrow fallback.
//!
//! Floating-point values use the same total ordering as [`Scalar`]: negative NaNs sort below
//! negative infinity, positive NaNs sort above positive infinity, and `-0.0 < +0.0`. Equality is
//! bitwise, so signed zeros and distinct NaN payloads remain distinct.
//!
//! [`Compare`] binds a [`CompareOperator`] to two operands. It can execute through
//! encoding-specific kernels before reaching the native dispatcher. [`CompareExecuteAdaptor`]
//! connects these execution paths to [`CompareKernel`], which lets an encoding compare values
//! without first converting them to canonical arrays.
//!
//! [`RowFn`]: crate::scalar_fn::unstable::row::RowFn

use std::cmp::Ordering;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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
use crate::arrays::ScalarFnArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::BoundExpression;
use crate::expr::bound;
use crate::expr::display::ExprDisplay;
use crate::kernel::ExecuteParentKernel;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::binary::binary_child_name;
use crate::scalar_fn::builtin::binary::execute_strict_step;
use crate::scalar_fn::builtin::binary::fmt_infix;
use crate::scalar_fn::builtin::literal::Literal;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::validity::Validity;

mod boolean;
mod bytes;
mod decimal;

mod nested;
pub use nested::RowComparator;
pub use nested::build_comparator as build_row_comparator;

mod primitive;

/// A comparison of two operands with a [`CompareOperator`].
///
/// The result is null where an operand is null.
#[derive(Clone)]
pub struct Compare;

impl Compare {
    /// Creates a lazy comparison of `lhs` and `rhs`.
    ///
    /// The operands must have the same logical dtype, ignoring nullability, even when they are
    /// empty or null. Extension dtypes must match, including their metadata; storage dtypes are
    /// distinct.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(
        lhs: ArrayRef,
        rhs: ArrayRef,
        operator: CompareOperator,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Compare.bind(operator), [lhs, rhs])
    }
}

impl ScalarFnVTable for Compare {
    type Options = CompareOperator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.compare");
        *ID
    }

    fn serialize(&self, operator: &CompareOperator) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CompareOpts {
                op: pb::compare_opts::CompareOp::from(*operator).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<CompareOperator> {
        let opts = pb::CompareOpts::decode(metadata)?;

        Ok(pb::compare_opts::CompareOp::try_from(opts.op)?.into())
    }

    fn arity(&self, _operator: &CompareOperator) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _operator: &CompareOperator, child_idx: usize) -> ChildName {
        binary_child_name(child_idx)
    }

    fn fmt_sql(
        &self,
        operator: &CompareOperator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        fmt_infix(operator, expr, f)
    }

    fn return_dtype(
        &self,
        _operator: &CompareOperator,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];
        validate_comparison_dtypes(lhs, rhs)?;

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        operator: &CompareOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        execute_compare(&lhs, &rhs, *operator, ctx)
    }

    fn simplify(
        &self,
        _operator: &CompareOperator,
        expr: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        let is_literal_null =
            |expr: &BoundExpression| expr.as_opt::<Literal>().is_some_and(Scalar::is_null);

        // Binding already rejected ill-typed comparisons such as `int_col = null_utf8`, so a
        // comparison against a null literal always evaluates to null.
        if is_literal_null(expr.child(0)) || is_literal_null(expr.child(1)) {
            return Ok(Some(bound::lit(Scalar::null(DType::Bool(
                Nullability::Nullable,
            )))));
        }

        Ok(None)
    }

    fn execute_step(
        &self,
        operator: &CompareOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        execute_strict_step(self, operator, args, ctx)
    }

    fn is_strict(&self, _operator: &CompareOperator) -> bool {
        true
    }

    fn is_infallible(&self, _operator: &CompareOperator) -> bool {
        true
    }
}

/// Trait for encoding-specific comparison kernels that operate in encoded space.
///
/// Implementations can compare an encoded array against another array (typically a constant)
/// without first decompressing. The adaptor normalizes operand order so `array` is always
/// the left-hand side, swapping the operator when necessary.
pub trait CompareKernel: VTable {
    /// Compares `lhs` with `rhs`, or returns `None` to use the default comparison path.
    ///
    /// Operand dtypes must match, ignoring nullability. [`CompareExecuteAdaptor`] checks this before
    /// dispatch; direct callers must satisfy the same requirement.
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that bridges [`CompareKernel`] implementations to [`ExecuteParentKernel`].
///
/// When a `ScalarFnArray(Compare, cmp_op)` wraps a child that implements `CompareKernel`,
/// this adaptor extracts the comparison operator and other operand, normalizes operand order
/// (swapping the operator if the encoded array is on the RHS), and delegates to the kernel.
#[derive(Default, Debug)]
pub struct CompareExecuteAdaptor<V>(
    /// The encoding whose comparison kernel handles the normalized operands.
    pub V,
);

impl<V> ExecuteParentKernel<V> for CompareExecuteAdaptor<V>
where
    V: CompareKernel,
{
    type Parent = ExactScalarFn<Compare>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Compare>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let cmp_op = *parent.options;

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
/// This is the entry point for compare operations from the [`Compare`] function.
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

/// Dispatches operands with matching logical dtypes to their native comparison kernel.
///
/// Extension operands are unwrapped before dispatch to the storage kernel.
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
        DType::Primitive(..) => primitive::compare_primitive(lhs, rhs, op, ctx),
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
/// Extension constants retain their constant encoding. Other extension arrays execute to their
/// canonical representation so that the caller can access their storage.
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

    // Some values, such as variants, have no scalar ordering even when their dtypes match.
    let ordering = lhs.partial_cmp(rhs).ok_or_else(|| {
        vortex_err!(
            "Scalar comparison requires comparable values, got {} and {}",
            lhs.dtype(),
            rhs.dtype()
        )
    })?;

    let result = match operator {
        CompareOperator::Eq => ordering.is_eq(),
        CompareOperator::NotEq => ordering.is_ne(),
        CompareOperator::Gt => ordering.is_gt(),
        CompareOperator::Gte => ordering.is_ge(),
        CompareOperator::Lt => ordering.is_lt(),
        CompareOperator::Lte => ordering.is_le(),
    };

    Ok(Scalar::bool(result, nullability))
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

#[cfg(test)]
mod tests;
