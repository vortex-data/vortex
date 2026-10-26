// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

#[expect(deprecated)]
pub use boolean::and_kleene;
#[expect(deprecated)]
pub use boolean::or_kleene;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::proto::expr as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::scalar_fn::builtin::operators::Operator;
use crate::scalar_fn::display::ExprDisplay;

pub mod boolean;
pub use boolean::BooleanExecuteAdaptor;
pub use boolean::BooleanKernel;
pub(crate) use boolean::execute_boolean;
pub use boolean::kleene_boolean_buffer_scalar;
pub use boolean::kleene_boolean_buffers;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;
mod primitive_operand;

use crate::scalar::NumericOperator;
use crate::scalar::Scalar;

#[derive(Clone)]
pub struct Binary;

impl Binary {
    /// Creates a lazy binary operation over `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(
        lhs: ArrayRef,
        rhs: ArrayRef,
        operator: Operator,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Binary.bind(operator), [lhs, rhs])
    }
}

/// Kleene and/or lookup table where both arguments are constant and non-NULL.
/// is_and, left, right
const KLEENE_LUT: [[[bool; 2]; 2]; 2] = [
    [[false, true], [true, true]],   // or
    [[false, false], [false, true]], // and
];

/// Kleene and/or reduction where one argument is constant non-NULL and other
/// is non-constant.
fn kleene_one_const<T: ReduceNode>(node: T, constant: bool, is_and: bool) -> T {
    match (is_and, constant) {
        (true, true) | (false, false) => node,
        (is_and, constant) => node.new_constant((constant && !is_and).into()),
    }
}

impl ScalarFnVTable for Binary {
    type Options = Operator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.binary");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BinaryOpts {
                op: (*instance).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::BinaryOpts::decode(_metadata)?;
        Operator::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Binary has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &Operator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " {} ", operator)?;
        Display::fmt(expr.display_child(1), f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, operator: &Operator, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if operator.is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }

            if let DType::Decimal(decimal_dtype, _) = lhs
                && lhs.eq_ignore_nullability(rhs)
            {
                let numeric_op = NumericOperator::try_from(*operator)?;
                return Ok(DType::Decimal(
                    numeric_op_result_decimal_dtype(*decimal_dtype, numeric_op)?,
                    lhs.nullability() | rhs.nullability(),
                ));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        if operator.is_comparison()
            && !lhs.eq_ignore_nullability(rhs)
            && !lhs.is_extension()
            && !rhs.is_extension()
        {
            vortex_bail!("Cannot compare different DTypes {} and {}", lhs, rhs);
        }

        if matches!(operator, Operator::And | Operator::Or)
            && !(lhs.is_boolean() && rhs.is_boolean())
        {
            vortex_bail!("'{operator}' requires bool operands, got {lhs} and {rhs}",);
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        op: &Operator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        match op {
            Operator::Eq => execute_compare(&lhs, &rhs, CompareOperator::Eq, ctx),
            Operator::NotEq => execute_compare(&lhs, &rhs, CompareOperator::NotEq, ctx),
            Operator::Lt => execute_compare(&lhs, &rhs, CompareOperator::Lt, ctx),
            Operator::Lte => execute_compare(&lhs, &rhs, CompareOperator::Lte, ctx),
            Operator::Gt => execute_compare(&lhs, &rhs, CompareOperator::Gt, ctx),
            Operator::Gte => execute_compare(&lhs, &rhs, CompareOperator::Gte, ctx),
            Operator::And => execute_boolean(lhs, rhs, Operator::And, ctx),
            Operator::Or => execute_boolean(lhs, rhs, Operator::Or, ctx),
            Operator::Add => execute_numeric(&lhs, &rhs, NumericOperator::Add, ctx),
            Operator::Sub => execute_numeric(&lhs, &rhs, NumericOperator::Sub, ctx),
            Operator::Mul => execute_numeric(&lhs, &rhs, NumericOperator::Mul, ctx),
            Operator::Div => execute_numeric(&lhs, &rhs, NumericOperator::Div, ctx),
        }
    }

    fn simplify<T: ReduceNode>(&self, operator: &Operator, node: &T) -> VortexResult<Option<T>> {
        let lhs = node.child(0);
        let rhs = node.child(1);

        let is_constant_null = |node: &T| node.as_constant().is_some_and(|s| s.is_null());

        // Binding already rejected ill-typed comparisons such as `int_col = null_utf8`, so a
        // comparison against a null constant always evaluates to null.
        if operator.is_comparison() && (is_constant_null(&lhs) || is_constant_null(&rhs)) {
            return Ok(Some(
                node.new_constant(Scalar::null(DType::Bool(Nullability::Nullable))),
            ));
        }

        if !matches!(operator, Operator::And | Operator::Or) {
            return Ok(None);
        }

        let bool_constant = |node: &T| node.as_constant()?.as_bool_opt().map(|value| value.value());

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
        // above (`null AND true`, `null OR false`) or cannot be reduced under
        // Kleene semantics (`null AND x`, `null OR x` for non-constant `x`).
        // Reduction must keep the result nullability binding inferred: the annihilator
        // constant and the identity child can both be less nullable than `lhs OP rhs`.
        let result_nullable = lhs.node_dtype()?.is_nullable() || rhs.node_dtype()?.is_nullable();
        let nullability = Nullability::from(result_nullable);
        let bool_constant_node = |value: bool| node.new_constant(Scalar::bool(value, nullability));
        let widen = |child: T| -> VortexResult<T> {
            if child.node_dtype()?.is_nullable() == result_nullable {
                Ok(child)
            } else {
                child.new_node(
                    Cast.bind(DType::Bool(nullability)),
                    std::slice::from_ref(&child),
                )
            }
        };

        Ok(match operator {
            Operator::And => match (bool_constant(&lhs), bool_constant(&rhs)) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => Some(bool_constant_node(false)),
                (Some(Some(true)), _) => Some(widen(rhs)?),
                (_, Some(Some(true))) => Some(widen(lhs)?),
                (Some(None), Some(None)) => Some(widen(lhs)?),
                _ => None,
            },
            Operator::Or => match (bool_constant(&lhs), bool_constant(&rhs)) {
                (Some(Some(true)), _) | (_, Some(Some(true))) => Some(bool_constant_node(true)),
                (Some(Some(false)), _) => Some(widen(rhs)?),
                (_, Some(Some(false))) => Some(widen(lhs)?),
                (Some(None), Some(None)) => Some(widen(lhs)?),
                _ => None,
            },
            _ => None,
        })
    }

    fn reduce<T: ReduceNode>(&self, operator: &Operator, node: &T) -> VortexResult<Option<T>> {
        if !matches!(operator, Operator::And | Operator::Or) {
            return Ok(None);
        }
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
        let is_and = *operator == Operator::And;

        Ok(Some(match (left_const, right_const) {
            (None, None) => return Ok(None),
            (Some(left_const), Some(right_const)) => left.new_constant(
                KLEENE_LUT[is_and as usize][left_const as usize][right_const as usize].into(),
            ),
            (Some(constant), None) => kleene_one_const(right, constant, is_and),
            (None, Some(constant)) => kleene_one_const(left, constant, is_and),
        }))
    }

    fn is_strict(&self, operator: &Operator) -> bool {
        !matches!(operator, Operator::And | Operator::Or)
    }

    fn is_infallible(&self, operator: &Operator) -> bool {
        // Arithmetic operations could be better modelled here.
        matches!(
            operator,
            Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte
                | Operator::And
                | Operator::Or
        )
    }
}
