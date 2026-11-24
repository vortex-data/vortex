// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Kleene boolean scalar function and its encoding-specific execution adaptor.
//!
//! [`Boolean`] binds a [`BooleanOperator`] to two boolean operands. Execution can use
//! encoding-specific kernels through [`BooleanExecuteAdaptor`] or the native kernel.

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::BoundExpression;
use crate::expr::bound;
use crate::expr::display::ExprDisplay;
use crate::matcher::Matcher;
use crate::proto::expr as pb;
use crate::require_input;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::binary::binary_child_name;
use crate::scalar_fn::builtin::binary::fmt_infix;
use crate::scalar_fn::builtin::literal::Literal;
use crate::scalar_fn::builtin::operators::BooleanOperator;

mod kernel;
pub use kernel::BooleanExecuteAdaptor;
pub use kernel::BooleanKernel;
#[expect(deprecated)]
pub use kernel::and_kleene;
pub use kernel::kleene_boolean_buffer_scalar;
pub use kernel::kleene_boolean_buffers;
#[expect(deprecated)]
pub use kernel::or_kleene;

/// A Kleene boolean operation over two boolean operands with a [`BooleanOperator`].
#[derive(Clone)]
pub struct Boolean;

impl Boolean {
    /// Creates a lazy boolean operation over `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or are not boolean.
    pub fn try_new(
        lhs: ArrayRef,
        rhs: ArrayRef,
        operator: BooleanOperator,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Boolean.bind(operator), [lhs, rhs])
    }
}

/// Kleene and/or lookup table where both arguments are constant and non-NULL.
/// is_and, left, right
const KLEENE_LUT: [[[bool; 2]; 2]; 2] = [
    [[false, true], [true, true]],   // or
    [[false, false], [false, true]], // and
];

impl ScalarFnVTable for Boolean {
    type Options = BooleanOperator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.boolean");
        *ID
    }

    fn serialize(&self, operator: &BooleanOperator) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BooleanOpts {
                op: pb::boolean_opts::BooleanOp::from(*operator).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<BooleanOperator> {
        let opts = pb::BooleanOpts::decode(metadata)?;

        Ok(pb::boolean_opts::BooleanOp::try_from(opts.op)?.into())
    }

    fn arity(&self, _operator: &BooleanOperator) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _operator: &BooleanOperator, child_idx: usize) -> ChildName {
        binary_child_name(child_idx)
    }

    fn fmt_sql(
        &self,
        operator: &BooleanOperator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        fmt_infix(operator, expr, f)
    }

    fn return_dtype(
        &self,
        operator: &BooleanOperator,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if !(lhs.is_boolean() && rhs.is_boolean()) {
            vortex_bail!("'{operator}' requires bool operands, got {lhs} and {rhs}",);
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        operator: &BooleanOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        kernel::execute_boolean(lhs, rhs, *operator, ctx)
    }

    fn simplify(
        &self,
        operator: &BooleanOperator,
        expr: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        let bool_literal = |expr: &BoundExpression| {
            expr.as_opt::<Literal>()?
                .as_bool_opt()
                .map(|value| value.value())
        };

        // AND/OR use Kleene three-valued logic. `None` below is a boolean null.
        //
        // AND:
        // - false AND x => false
        // - true  AND x => x
        // - null  AND null => null
        //
        // OR:
        // - true  OR x => true
        // - false OR x => x
        // - null  OR null => null
        //
        // Other null cases either fall out of the identity/annihilator rules
        // above (`null AND true`, `null OR false`) or cannot be simplified under
        // Kleene semantics (`null AND x`, `null OR x` for non-literal `x`).
        // Simplification must keep the result nullability binding inferred: the annihilator
        // literal and the identity child can both be less nullable than `lhs OP rhs`.
        let result_nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
        let nullability = Nullability::from(result_nullable);
        let bool_lit = |value: bool| bound::lit(Scalar::bool(value, nullability));
        let widen = |child: &BoundExpression| {
            if child.dtype().is_nullable() == result_nullable {
                child.clone()
            } else {
                bound::cast(child.clone(), DType::Bool(nullability))
            }
        };

        Ok(match operator {
            BooleanOperator::And => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => Some(bool_lit(false)),
                (Some(Some(true)), _) => Some(widen(rhs)),
                (_, Some(Some(true))) => Some(widen(lhs)),
                (Some(None), Some(None)) => Some(widen(lhs)),
                _ => None,
            },
            BooleanOperator::Or => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(true)), _) | (_, Some(Some(true))) => Some(bool_lit(true)),
                (Some(Some(false)), _) => Some(widen(rhs)),
                (_, Some(Some(false))) => Some(widen(lhs)),
                (Some(None), Some(None)) => Some(widen(lhs)),
                _ => None,
            },
        })
    }

    fn reduce<T: ReduceNode>(
        &self,
        operator: &BooleanOperator,
        node: &T,
    ) -> VortexResult<Option<T>> {
        let left = node.child(0);
        let right = node.child(1);

        let left_const = left.as_constant();
        let right_const = right.as_constant();

        // We don't handle Kleene NULL reduction here (.value() returns None
        // for NULL). This will be reduced in the boolean kernel during
        // execution. Not handling the case keeps the code much simpler.
        let left_const = left_const
            .and_then(|s| s.value().cloned())
            .map(|v| v.as_bool());
        let right_const = right_const
            .and_then(|s| s.value().cloned())
            .map(|v| v.as_bool());
        let is_and = *operator == BooleanOperator::And;
        let nullability = node.node_dtype()?.nullability();

        let (constant, other) = match (left_const, right_const) {
            (None, None) => return Ok(None),
            (Some(left_const), Some(right_const)) => {
                return Ok(Some(left.new_constant(Scalar::bool(
                    KLEENE_LUT[is_and as usize][left_const as usize][right_const as usize],
                    nullability,
                ))));
            }
            (Some(constant), None) => (constant, right),
            (None, Some(constant)) => (constant, left),
        };

        // `x AND true` and `x OR false` are `x`, but only when `x` already has the result's
        // nullability: otherwise leave the cast to the boolean kernel.
        if constant == is_and {
            return Ok((other.node_dtype()?.nullability() == nullability).then_some(other));
        }
        // `x AND false` and `x OR true` are the constant itself.
        Ok(Some(
            other.new_constant(Scalar::bool(constant, nullability)),
        ))
    }

    fn execute_step(
        &self,
        operator: &BooleanOperator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        let columnar = |idx| args.input_matches(idx, AnyColumnar::matches);
        let constant = |idx| args.input_matches(idx, |array| array.is::<Constant>());

        // A constant operand can decide the result, such as `false AND x`, so `execute` does not
        // need the other operand executed.
        let ready = (columnar(0)? && columnar(1)?) || constant(0)? || constant(1)?;
        if !ready {
            require_input!(args, 0 => AnyColumnar);
            require_input!(args, 1 => AnyColumnar);
        }

        self.execute(operator, args, ctx)
            .map(ScalarFnExecutionResult::done)
    }

    fn is_strict(&self, _operator: &BooleanOperator) -> bool {
        false
    }

    fn is_infallible(&self, _operator: &BooleanOperator) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
