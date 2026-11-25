// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The arithmetic scalar function and its native execution. There is no Arrow fallback.
//!
//! [`Numeric`] binds a [`NumericOperator`] (Add/Sub/Mul/Div) to two operands of the same
//! primitive or decimal dtype.
//!
//! The primitive widths are computed by a
//! [`RowFn`](crate::scalar_fn::unstable::row::RowFn), which owns null handling, constants, and
//! validity for them; see the `row` module. Decimal keeps its own columnar implementation in the
//! `decimal` module.

mod checked;
mod decimal;
mod primitive;
mod row;

use std::fmt::Formatter;

use decimal::execute_numeric_decimal;
use prost::Message;
use row::execute_numeric_primitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::proto::expr as pb;
use crate::scalar::NumericOperator;
pub(crate) use crate::scalar::decimal_numeric_result_dtype as numeric_op_result_decimal_dtype;
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
use crate::scalar_fn::display::ExprDisplay;

/// An arithmetic operation over two operands with a [`NumericOperator`].
///
/// The result is null where an operand is null. An operation fails when a result overflows, and
/// integer division fails when the divisor is zero.
#[derive(Clone)]
pub struct Numeric;

impl Numeric {
    /// Creates a lazy arithmetic operation over `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(
        lhs: ArrayRef,
        rhs: ArrayRef,
        operator: NumericOperator,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Numeric.bind(operator), [lhs, rhs])
    }
}

impl ScalarFnVTable for Numeric {
    type Options = NumericOperator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.numeric");
        *ID
    }

    fn serialize(&self, operator: &NumericOperator) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::NumericOpts {
                op: pb::numeric_opts::NumericOp::from(*operator).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<NumericOperator> {
        let opts = pb::NumericOpts::decode(metadata)?;

        Ok(pb::numeric_opts::NumericOp::try_from(opts.op)?.into())
    }

    fn arity(&self, _operator: &NumericOperator) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _operator: &NumericOperator, child_idx: usize) -> ChildName {
        binary_child_name(child_idx)
    }

    fn fmt_sql(
        &self,
        operator: &NumericOperator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        fmt_infix(operator, expr, f)
    }

    fn return_dtype(
        &self,
        operator: &NumericOperator,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
            return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
        }

        if let DType::Decimal(decimal_dtype, _) = lhs
            && lhs.eq_ignore_nullability(rhs)
        {
            return Ok(DType::Decimal(
                numeric_op_result_decimal_dtype(*decimal_dtype, *operator)?,
                lhs.nullability() | rhs.nullability(),
            ));
        }

        vortex_bail!(
            "incompatible types for arithmetic operation: {} {}",
            lhs,
            rhs
        );
    }

    fn execute(
        &self,
        operator: &NumericOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        execute_numeric(&lhs, &rhs, *operator, ctx)
    }

    fn execute_step(
        &self,
        operator: &NumericOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        execute_strict_step(self, operator, args, ctx)
    }

    fn is_strict(&self, _operator: &NumericOperator) -> bool {
        true
    }
}

/// Execute a numeric operation between two arrays.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        lhs.dtype().eq_ignore_nullability(rhs.dtype()),
        "numeric operator requires matching types, got {} and {}",
        lhs.dtype(),
        rhs.dtype()
    );

    let dtype = lhs.dtype();
    vortex_ensure!(
        matches!(dtype, DType::Primitive(..) | DType::Decimal(..)),
        "numeric operator is not supported for dtype {}",
        dtype
    );

    vortex_ensure_eq!(
        lhs.len(),
        rhs.len(),
        "numeric operator requires equal lengths"
    );

    if lhs.is_empty() {
        return build_empty_result(lhs, rhs, op);
    }

    match dtype {
        DType::Primitive(..) => execute_numeric_primitive(lhs, rhs, op, ctx),
        DType::Decimal(..) => execute_numeric_decimal(lhs, rhs, op, ctx),
        _ => unreachable!("dtype is either Primitive or Decimal"),
    }
}

fn build_empty_result(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<ArrayRef> {
    let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
    let result_dtype = match lhs.dtype() {
        DType::Primitive(..) => lhs.dtype().with_nullability(nullability),
        DType::Decimal(decimal_dtype, _) => DType::Decimal(
            numeric_op_result_decimal_dtype(*decimal_dtype, op)?,
            nullability,
        ),
        _ => unreachable!("dtype is either Primitive or Decimal"),
    };

    Ok(Canonical::empty(&result_dtype).into_array())
}

#[cfg(test)]
mod tests;
