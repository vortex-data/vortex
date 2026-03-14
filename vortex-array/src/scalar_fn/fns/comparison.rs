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
use vortex_proto::expr::comparison_opts::ComparisonOp as PbComparisonOp;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::and_collect;
use crate::expr::eq;
use crate::expr::expression::Expression;
use crate::expr::gt;
use crate::expr::gt_eq;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::lt_eq;
use crate::expr::or_collect;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::execute_compare;
use crate::scalar_fn::fns::operators::Operator;

/// The six comparison operators, providing compile-time guarantees that only
/// comparison variants are used where comparisons are expected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareOperator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another.
    Gt,
    /// Expression is greater or equal to another.
    Gte,
    /// Expression is less than another.
    Lt,
    /// Expression is less or equal to another.
    Lte,
}

impl CompareOperator {
    /// Return the logical inverse of this comparison operator.
    pub fn inverse(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::NotEq,
            CompareOperator::NotEq => CompareOperator::Eq,
            CompareOperator::Gt => CompareOperator::Lte,
            CompareOperator::Gte => CompareOperator::Lt,
            CompareOperator::Lt => CompareOperator::Gte,
            CompareOperator::Lte => CompareOperator::Gt,
        }
    }

    /// Swap the sides of the operator so that swapping lhs and rhs preserves the result.
    pub fn swap(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::Eq,
            CompareOperator::NotEq => CompareOperator::NotEq,
            CompareOperator::Gt => CompareOperator::Lt,
            CompareOperator::Gte => CompareOperator::Lte,
            CompareOperator::Lt => CompareOperator::Gt,
            CompareOperator::Lte => CompareOperator::Gte,
        }
    }
}

impl Display for CompareOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            CompareOperator::Eq => "=",
            CompareOperator::NotEq => "!=",
            CompareOperator::Gt => ">",
            CompareOperator::Gte => ">=",
            CompareOperator::Lt => "<",
            CompareOperator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl From<CompareOperator> for Operator {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => Operator::Eq,
            CompareOperator::NotEq => Operator::NotEq,
            CompareOperator::Gt => Operator::Gt,
            CompareOperator::Gte => Operator::Gte,
            CompareOperator::Lt => Operator::Lt,
            CompareOperator::Lte => Operator::Lte,
        }
    }
}

impl TryFrom<Operator> for CompareOperator {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        match value {
            Operator::Eq => Ok(CompareOperator::Eq),
            Operator::NotEq => Ok(CompareOperator::NotEq),
            Operator::Gt => Ok(CompareOperator::Gt),
            Operator::Gte => Ok(CompareOperator::Gte),
            Operator::Lt => Ok(CompareOperator::Lt),
            Operator::Lte => Ok(CompareOperator::Lte),
            other => Err(vortex_error::vortex_err!(
                InvalidArgument: "{other} is not a comparison operator"
            )),
        }
    }
}

impl From<CompareOperator> for PbComparisonOp {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => PbComparisonOp::Eq,
            CompareOperator::NotEq => PbComparisonOp::NotEq,
            CompareOperator::Gt => PbComparisonOp::Gt,
            CompareOperator::Gte => PbComparisonOp::Gte,
            CompareOperator::Lt => PbComparisonOp::Lt,
            CompareOperator::Lte => PbComparisonOp::Lte,
        }
    }
}

impl From<CompareOperator> for i32 {
    fn from(value: CompareOperator) -> Self {
        let op: PbComparisonOp = value.into();
        op.into()
    }
}

impl From<PbComparisonOp> for CompareOperator {
    fn from(value: PbComparisonOp) -> Self {
        match value {
            PbComparisonOp::Eq => CompareOperator::Eq,
            PbComparisonOp::NotEq => CompareOperator::NotEq,
            PbComparisonOp::Gt => CompareOperator::Gt,
            PbComparisonOp::Gte => CompareOperator::Gte,
            PbComparisonOp::Lt => CompareOperator::Lt,
            PbComparisonOp::Lte => CompareOperator::Lte,
        }
    }
}

impl TryFrom<i32> for CompareOperator {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(PbComparisonOp::try_from(value)?.into())
    }
}

/// A comparison scalar function for the six relational operators.
#[derive(Clone)]
pub struct Comparison;

impl ScalarFnVTable for Comparison {
    type Options = CompareOperator;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.comparison")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::ComparisonOpts {
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
        let opts = pb::ComparisonOpts::decode(metadata)?;
        CompareOperator::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Comparison has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &CompareOperator,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} ", operator)?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(
        &self,
        _operator: &CompareOperator,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if !lhs.eq_ignore_nullability(rhs) && !lhs.is_extension() && !rhs.is_extension() {
            vortex_bail!("Cannot compare different DTypes {} and {}", lhs, rhs);
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        op: &CompareOperator,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        execute_compare(&lhs, &rhs, *op)
    }

    fn stat_falsification(
        &self,
        operator: &CompareOperator,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        #[inline]
        fn with_nan_predicate(
            lhs: &Expression,
            rhs: &Expression,
            value_predicate: Expression,
            catalog: &dyn StatsCatalog,
        ) -> Expression {
            let nan_predicate = and_collect(
                lhs.stat_expression(Stat::NaNCount, catalog)
                    .into_iter()
                    .chain(rhs.stat_expression(Stat::NaNCount, catalog))
                    .map(|nans| eq(nans, lit(0u64))),
            );

            if let Some(nan_check) = nan_predicate {
                and(nan_check, value_predicate)
            } else {
                value_predicate
            }
        }

        let lhs = expr.child(0);
        let rhs = expr.child(1);
        match operator {
            CompareOperator::Eq => {
                let min_lhs = lhs.stat_min(catalog);
                let max_lhs = lhs.stat_max(catalog);
                let min_rhs = rhs.stat_min(catalog);
                let max_rhs = rhs.stat_max(catalog);

                let left = min_lhs.zip(max_rhs).map(|(a, b)| gt(a, b));
                let right = min_rhs.zip(max_lhs).map(|(a, b)| gt(a, b));

                let min_max_check = or_collect(left.into_iter().chain(right))?;
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            CompareOperator::NotEq => {
                let min_lhs = lhs.stat_min(catalog)?;
                let max_lhs = lhs.stat_max(catalog)?;
                let min_rhs = rhs.stat_min(catalog)?;
                let max_rhs = rhs.stat_max(catalog)?;

                let min_max_check = and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs));
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            CompareOperator::Gt => {
                let min_max_check = lt_eq(lhs.stat_max(catalog)?, rhs.stat_min(catalog)?);
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            CompareOperator::Gte => {
                let min_max_check = lt(lhs.stat_max(catalog)?, rhs.stat_min(catalog)?);
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            CompareOperator::Lt => {
                let min_max_check = gt_eq(lhs.stat_min(catalog)?, rhs.stat_max(catalog)?);
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            CompareOperator::Lte => {
                let min_max_check = gt(lhs.stat_min(catalog)?, rhs.stat_max(catalog)?);
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
        }
    }

    fn validity(
        &self,
        _operator: &CompareOperator,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let lhs = expression.child(0).validity()?;
        let rhs = expression.child(1).validity()?;
        Ok(Some(and(lhs, rhs)))
    }

    fn is_null_sensitive(&self, _operator: &CompareOperator) -> bool {
        false
    }

    fn is_fallible(&self, _operator: &CompareOperator) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::CompareOperator;
    use super::Comparison;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_return_dtype() {
        use crate::expr::Expression;
        use crate::expr::col;
        use crate::expr::test_harness;

        let dtype = test_harness::struct_dtype();
        let col1: Expression = col("col1");
        let col2: Expression = col("col2");

        let eq_expr = Comparison.new_expr(CompareOperator::Eq, [col1, col2]);
        assert_eq!(
            eq_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    fn test_display() {
        use crate::expr::col;

        let expr = Comparison.new_expr(CompareOperator::Gt, [col("a"), col("b")]);
        assert_eq!(format!("{expr}"), "($.a > $.b)");
    }

    #[test]
    fn test_constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let result = left
            .into_array()
            .binary(right.into_array(), Operator::Gt)
            .unwrap();
        assert_eq!(result.len(), 10);
        let scalar = result.scalar_at(0).unwrap();
        assert_eq!(scalar.as_bool().value(), Some(false));
    }

    #[test]
    fn test_struct_comparison() {
        use crate::arrays::PrimitiveArray;

        let lhs_struct = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32]).into_array()),
            ("b", PrimitiveArray::from_iter([3i32]).into_array()),
        ])
        .unwrap()
        .into_array();

        let rhs_struct_equal = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32]).into_array()),
            ("b", PrimitiveArray::from_iter([3i32]).into_array()),
        ])
        .unwrap()
        .into_array();

        let result = lhs_struct.binary(rhs_struct_equal, Operator::Eq).unwrap();
        assert_eq!(
            result.scalar_at(0).vortex_expect("value"),
            Scalar::bool(true, Nullability::NonNullable),
        );
    }

    #[test]
    fn test_varbin_compare() {
        use crate::arrays::VarBinArray;
        use crate::arrays::VarBinViewArray;

        let left = VarBinArray::from(vec!["a", "b"]).into_array();
        let right = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        let res = left.binary(right, Operator::Eq).unwrap();
        assert_arrays_eq!(res, BoolArray::from_iter([true, true]));
    }

    #[test]
    fn test_scalar_compare() {
        let values = buffer![1i32, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(2i32), 3).into_array();
        let result = values.binary(rhs, Operator::Gt).unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]));
    }
}
