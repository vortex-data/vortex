// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::expression::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::arithmetic::Arithmetic;
use crate::scalar_fn::fns::arithmetic::ArithmeticOp;
use crate::scalar_fn::fns::comparison::Comparison;
use crate::scalar_fn::fns::logical::LogicalBinary;
use crate::scalar_fn::fns::logical::LogicalOp;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::vtable::ReduceCtx;
use crate::scalar_fn::vtable::ReduceNode;
use crate::scalar_fn::vtable::ReduceNodeRef;

pub(crate) mod boolean;
pub(crate) use boolean::*;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;

#[derive(Clone)]
pub struct Binary;

impl ScalarFnVTable for Binary {
    type Options = Operator;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.binary")
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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} ", operator)?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, operator: &Operator, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match operator {
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                Arithmetic.return_dtype(&ArithmeticOp::try_from(*operator)?, arg_dtypes)
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Gt
            | Operator::Gte
            | Operator::Lt
            | Operator::Lte => {
                Comparison.return_dtype(&CompareOperator::try_from(*operator)?, arg_dtypes)
            }
            Operator::And | Operator::Or => {
                LogicalBinary.return_dtype(&LogicalOp::try_from(*operator)?, arg_dtypes)
            }
        }
    }

    fn execute(
        &self,
        op: &Operator,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "Binary scalar function must be reduced before execution: {:?}",
            op
        );
    }

    fn reduce(
        &self,
        operator: &Operator,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        let children = [node.child(0), node.child(1)];
        let scalar_fn = match operator {
            Operator::Eq => Comparison.bind(CompareOperator::Eq),
            Operator::NotEq => Comparison.bind(CompareOperator::NotEq),
            Operator::Gt => Comparison.bind(CompareOperator::Gt),
            Operator::Gte => Comparison.bind(CompareOperator::Gte),
            Operator::Lt => Comparison.bind(CompareOperator::Lt),
            Operator::Lte => Comparison.bind(CompareOperator::Lte),
            Operator::And => LogicalBinary.bind(LogicalOp::And),
            Operator::Or => LogicalBinary.bind(LogicalOp::Or),
            Operator::Add => Arithmetic.bind(ArithmeticOp::Add),
            Operator::Sub => Arithmetic.bind(ArithmeticOp::Sub),
            Operator::Mul => Arithmetic.bind(ArithmeticOp::Mul),
            Operator::Div => Arithmetic.bind(ArithmeticOp::Div),
        };
        Ok(Some(ctx.new_node(scalar_fn, &children)?))
    }
}

#[cfg(test)]
mod tests {
    use crate::expr::Expression;
    use crate::expr::and_collect;
    use crate::expr::lit;
    use crate::expr::or_collect;

    #[test]
    fn and_collect_balanced() {
        let values = vec![lit(1), lit(2), lit(3), lit(4), lit(5)];

        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(and)
        ├── lhs: vortex.binary(and)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(and)
            ├── lhs: vortex.binary(and)
            │   ├── lhs: vortex.literal(3i32)
            │   └── rhs: vortex.literal(4i32)
            └── rhs: vortex.literal(5i32)
        ");

        // 4 elements: and(and(1, 2), and(3, 4)) - perfectly balanced
        let values = vec![lit(1), lit(2), lit(3), lit(4)];
        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(and)
        ├── lhs: vortex.binary(and)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(and)
            ├── lhs: vortex.literal(3i32)
            └── rhs: vortex.literal(4i32)
        ");

        // 1 element: just the element
        let values = vec![lit(1)];
        insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @"vortex.literal(1i32)");

        // 0 elements: None
        let values: Vec<Expression> = vec![];
        assert!(and_collect(values.into_iter()).is_none());
    }

    #[test]
    fn or_collect_balanced() {
        // 4 elements: or(or(1, 2), or(3, 4)) - perfectly balanced
        let values = vec![lit(1), lit(2), lit(3), lit(4)];
        insta::assert_snapshot!(or_collect(values.into_iter()).unwrap().display_tree(), @r"
        vortex.binary(or)
        ├── lhs: vortex.binary(or)
        │   ├── lhs: vortex.literal(1i32)
        │   └── rhs: vortex.literal(2i32)
        └── rhs: vortex.binary(or)
            ├── lhs: vortex.literal(3i32)
            └── rhs: vortex.literal(4i32)
        ");
    }
}
