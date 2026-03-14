// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_proto::expr::arithmetic_opts::ArithmeticOp as PbArithmeticOp;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::and;
use crate::expr::expression::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::execute_numeric;
use crate::scalar_fn::fns::operators::Operator;

/// Arithmetic binary operators.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArithmeticOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

impl Display for ArithmeticOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            ArithmeticOp::Add => "+",
            ArithmeticOp::Sub => "-",
            ArithmeticOp::Mul => "*",
            ArithmeticOp::Div => "/",
        };
        Display::fmt(display, f)
    }
}

impl From<ArithmeticOp> for Operator {
    fn from(value: ArithmeticOp) -> Self {
        match value {
            ArithmeticOp::Add => Operator::Add,
            ArithmeticOp::Sub => Operator::Sub,
            ArithmeticOp::Mul => Operator::Mul,
            ArithmeticOp::Div => Operator::Div,
        }
    }
}

impl TryFrom<Operator> for ArithmeticOp {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        match value {
            Operator::Add => Ok(ArithmeticOp::Add),
            Operator::Sub => Ok(ArithmeticOp::Sub),
            Operator::Mul => Ok(ArithmeticOp::Mul),
            Operator::Div => Ok(ArithmeticOp::Div),
            other => Err(vortex_error::vortex_err!(
                InvalidArgument: "{other} is not an arithmetic operator"
            )),
        }
    }
}

impl From<ArithmeticOp> for PbArithmeticOp {
    fn from(value: ArithmeticOp) -> Self {
        match value {
            ArithmeticOp::Add => PbArithmeticOp::Add,
            ArithmeticOp::Sub => PbArithmeticOp::Sub,
            ArithmeticOp::Mul => PbArithmeticOp::Mul,
            ArithmeticOp::Div => PbArithmeticOp::Div,
        }
    }
}

impl From<ArithmeticOp> for i32 {
    fn from(value: ArithmeticOp) -> Self {
        let op: PbArithmeticOp = value.into();
        op.into()
    }
}

impl From<PbArithmeticOp> for ArithmeticOp {
    fn from(value: PbArithmeticOp) -> Self {
        match value {
            PbArithmeticOp::Add => ArithmeticOp::Add,
            PbArithmeticOp::Sub => ArithmeticOp::Sub,
            PbArithmeticOp::Mul => ArithmeticOp::Mul,
            PbArithmeticOp::Div => ArithmeticOp::Div,
        }
    }
}

impl TryFrom<i32> for ArithmeticOp {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(PbArithmeticOp::try_from(value)?.into())
    }
}

impl From<ArithmeticOp> for crate::scalar::NumericOperator {
    fn from(op: ArithmeticOp) -> Self {
        match op {
            ArithmeticOp::Add => crate::scalar::NumericOperator::Add,
            ArithmeticOp::Sub => crate::scalar::NumericOperator::Sub,
            ArithmeticOp::Mul => crate::scalar::NumericOperator::Mul,
            ArithmeticOp::Div => crate::scalar::NumericOperator::Div,
        }
    }
}

/// An arithmetic binary scalar function.
#[derive(Clone)]
pub struct Arithmetic;

impl ScalarFnVTable for Arithmetic {
    type Options = ArithmeticOp;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.arithmetic")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::ArithmeticOpts {
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
        let opts = pb::ArithmeticOpts::decode(metadata)?;
        ArithmeticOp::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Arithmetic has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &ArithmeticOp,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} ", operator)?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _operator: &ArithmeticOp, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
            return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
        }
        vortex_bail!(
            "incompatible types for arithmetic operation: {} {}",
            lhs,
            rhs
        );
    }

    fn execute(
        &self,
        op: &ArithmeticOp,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        execute_numeric(&lhs, &rhs, (*op).into())
    }

    fn validity(
        &self,
        _operator: &ArithmeticOp,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let lhs = expression.child(0).validity()?;
        let rhs = expression.child(1).validity()?;
        Ok(Some(and(lhs, rhs)))
    }

    fn is_null_sensitive(&self, _operator: &ArithmeticOp) -> bool {
        false
    }

    fn is_fallible(&self, _operator: &ArithmeticOp) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::Arithmetic;
    use super::ArithmeticOp;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_display() {
        use crate::expr::col;

        let expr = Arithmetic.new_expr(ArithmeticOp::Add, [col("a"), col("b")]);
        assert_eq!(format!("{expr}"), "($.a + $.b)");
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(1u16), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1i64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let rhs = ConstantArray::new(Scalar::from(Some(1u16)), 4).into_array();
        let result = values.into_array().binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1f64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_add() {
        let values = buffer![1i32, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(10i32), 3).into_array();
        let result = values.binary(rhs, Operator::Add).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([11i32, 12, 13]));
    }

    #[test]
    fn test_scalar_multiply() {
        let values = buffer![2i32, 3, 4].into_array();
        let rhs = ConstantArray::new(Scalar::from(3i32), 3).into_array();
        let result = values.binary(rhs, Operator::Mul).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([6i32, 9, 12]));
    }
}
