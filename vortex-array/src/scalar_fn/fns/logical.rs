// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;
use vortex_proto::expr::logical_binary_opts::LogicalBinaryOp as PbLogicalBinaryOp;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::expression::Expression;
use crate::expr::or_collect;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::execute_boolean;
use crate::scalar_fn::fns::operators::Operator;

/// Logical binary operators (Kleene three-valued logic).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LogicalOp {
    /// Boolean AND (∧) with Kleene logic.
    And,
    /// Boolean OR (∨) with Kleene logic.
    Or,
}

impl Display for LogicalOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            LogicalOp::And => "and",
            LogicalOp::Or => "or",
        };
        Display::fmt(display, f)
    }
}

impl From<LogicalOp> for Operator {
    fn from(value: LogicalOp) -> Self {
        match value {
            LogicalOp::And => Operator::And,
            LogicalOp::Or => Operator::Or,
        }
    }
}

impl TryFrom<Operator> for LogicalOp {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        match value {
            Operator::And => Ok(LogicalOp::And),
            Operator::Or => Ok(LogicalOp::Or),
            other => Err(vortex_error::vortex_err!(
                InvalidArgument: "{other} is not a logical operator"
            )),
        }
    }
}

impl From<LogicalOp> for PbLogicalBinaryOp {
    fn from(value: LogicalOp) -> Self {
        match value {
            LogicalOp::And => PbLogicalBinaryOp::And,
            LogicalOp::Or => PbLogicalBinaryOp::Or,
        }
    }
}

impl From<LogicalOp> for i32 {
    fn from(value: LogicalOp) -> Self {
        let op: PbLogicalBinaryOp = value.into();
        op.into()
    }
}

impl From<PbLogicalBinaryOp> for LogicalOp {
    fn from(value: PbLogicalBinaryOp) -> Self {
        match value {
            PbLogicalBinaryOp::And => LogicalOp::And,
            PbLogicalBinaryOp::Or => LogicalOp::Or,
        }
    }
}

impl TryFrom<i32> for LogicalOp {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(PbLogicalBinaryOp::try_from(value)?.into())
    }
}

/// A logical binary scalar function implementing Kleene three-valued logic.
#[derive(Clone)]
pub struct LogicalBinary;

impl ScalarFnVTable for LogicalBinary {
    type Options = LogicalOp;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.logical_binary")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LogicalBinaryOpts {
                op: (*options).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::LogicalBinaryOpts::decode(metadata)?;
        LogicalOp::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("LogicalBinary has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &LogicalOp,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} ", operator)?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _operator: &LogicalOp, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];
        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        op: &LogicalOp,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        let operator: Operator = (*op).into();
        execute_boolean(&lhs, &rhs, operator)
    }

    fn stat_falsification(
        &self,
        operator: &LogicalOp,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let lhs = expr.child(0);
        let rhs = expr.child(1);
        match operator {
            LogicalOp::And => or_collect(
                lhs.stat_falsification(catalog)
                    .into_iter()
                    .chain(rhs.stat_falsification(catalog)),
            ),
            LogicalOp::Or => Some(and(
                lhs.stat_falsification(catalog)?,
                rhs.stat_falsification(catalog)?,
            )),
        }
    }

    fn validity(
        &self,
        _operator: &LogicalOp,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // Kleene logic: validity cannot be determined without evaluating the expression.
        Ok(None)
    }

    fn is_null_sensitive(&self, _operator: &LogicalOp) -> bool {
        false
    }

    fn is_fallible(&self, _operator: &LogicalOp) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::LogicalBinary;
    use super::LogicalOp;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::Expression;
    use crate::expr::col;
    use crate::expr::test_harness;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::operators::Operator;

    #[rstest]
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = lhs.binary(rhs, Operator::Or).unwrap();
        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();
        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[rstest]
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = lhs.binary(rhs, Operator::And).unwrap();
        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();
        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[test]
    fn test_or_kleene_validity() {
        let struct_arr = StructArray::from_fields(&[
            ("a", BoolArray::from_iter([Some(true)]).into_array()),
            (
                "b",
                BoolArray::from_iter([Option::<bool>::None]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let expr = LogicalBinary.new_expr(LogicalOp::Or, [col("a"), col("b")]);
        let result = struct_arr.apply(&expr).unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([Some(true)]).into_array());
    }

    #[test]
    fn test_return_dtype() {
        let dtype = test_harness::struct_dtype();
        let bool1: Expression = col("bool1");
        let bool2: Expression = col("bool2");

        let and_expr = LogicalBinary.new_expr(LogicalOp::And, [bool1.clone(), bool2.clone()]);
        assert_eq!(
            and_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let or_expr = LogicalBinary.new_expr(LogicalOp::Or, [bool1, bool2]);
        assert_eq!(
            or_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = LogicalBinary.new_expr(LogicalOp::And, [col("a"), col("b")]);
        assert_eq!(format!("{expr}"), "($.a and $.b)");
    }
}
