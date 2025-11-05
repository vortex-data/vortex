// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::compute::{BetweenOptions, between as between_compute};
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::expression::Expression;
use crate::exprs::binary::Binary;
use crate::exprs::operators::Operator;
use crate::{ChildName, ExprId, ExpressionView, StatsCatalog, VTable, VTableExt};

/// An optimized scalar expression to compute whether values fall between two bounds.
///
/// This expression takes three children:
/// 1. The array of values to check.
/// 2. The lower bound.
/// 3. The upper bound.
///
/// The comparison strictness is controlled by the metadata.
///
/// NOTE: this expression will shortly be removed in favor of pipelined computation of two
/// separate comparisons combined with a logical AND.
pub struct Between;

impl VTable for Between {
    type Instance = BetweenOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.between")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BetweenOpts {
                lower_strict: instance.lower_strict.is_strict(),
                upper_strict: instance.upper_strict.is_strict(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let opts = pb::BetweenOpts::decode(metadata)?;
        Ok(Some(BetweenOptions {
            lower_strict: if opts.lower_strict {
                vortex_array::compute::StrictComparison::Strict
            } else {
                vortex_array::compute::StrictComparison::NonStrict
            },
            upper_strict: if opts.upper_strict {
                vortex_array::compute::StrictComparison::Strict
            } else {
                vortex_array::compute::StrictComparison::NonStrict
            },
        }))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 3 {
            vortex_bail!(
                "Between expression requires exactly 3 children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("array"),
            1 => ChildName::from("lower"),
            2 => ChildName::from("upper"),
            _ => unreachable!("Invalid child index {} for Between expression", child_idx),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        let options = expr.data();
        let lower_op = if options.lower_strict.is_strict() {
            "<"
        } else {
            "<="
        };
        let upper_op = if options.upper_strict.is_strict() {
            "<"
        } else {
            "<="
        };
        write!(
            f,
            "({} {} {} {} {})",
            expr.lower(),
            lower_op,
            expr.child(),
            upper_op,
            expr.upper()
        )
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let arr_dt = expr.child().return_dtype(scope)?;
        let lower_dt = expr.lower().return_dtype(scope)?;
        let upper_dt = expr.upper().return_dtype(scope)?;

        if !arr_dt.eq_ignore_nullability(&lower_dt) {
            vortex_bail!(
                "Array dtype {} does not match lower dtype {}",
                arr_dt,
                lower_dt
            );
        }
        if !arr_dt.eq_ignore_nullability(&upper_dt) {
            vortex_bail!(
                "Array dtype {} does not match upper dtype {}",
                arr_dt,
                upper_dt
            );
        }

        Ok(Bool(
            arr_dt.nullability() | lower_dt.nullability() | upper_dt.nullability(),
        ))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let arr = expr.child().evaluate(scope)?;
        let lower = expr.lower().evaluate(scope)?;
        let upper = expr.upper().evaluate(scope)?;
        between_compute(&arr, &lower, &upper, expr.data())
    }

    fn stat_falsification(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        expr.to_binary_expr().stat_falsification(catalog)
    }
}

impl ExpressionView<'_, Between> {
    pub fn child(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn lower(&self) -> &Expression {
        &self.children()[1]
    }

    pub fn upper(&self) -> &Expression {
        &self.children()[2]
    }

    pub fn to_binary_expr(&self) -> Expression {
        let options = self.data();
        let arr = self.children()[0].clone();
        let lower = self.children()[1].clone();
        let upper = self.children()[2].clone();

        let lhs = Binary.new_expr(
            options.lower_strict.to_operator().into(),
            [lower, arr.clone()],
        );
        let rhs = Binary.new_expr(options.upper_strict.to_operator().into(), [arr, upper]);
        Binary.new_expr(Operator::And, [lhs, rhs])
    }
}

/// Creates an expression that checks if values are between two bounds.
///
/// Returns a boolean array indicating which values fall within the specified range.
/// The comparison strictness is controlled by the options parameter.
///
/// ```rust
/// # use vortex_array::compute::BetweenOptions;
/// # use vortex_array::compute::StrictComparison;
/// # use vortex_expr::{between, lit, root};
/// let opts = BetweenOptions {
///     lower_strict: StrictComparison::NonStrict,
///     upper_strict: StrictComparison::NonStrict,
/// };
/// let expr = between(root(), lit(10), lit(20), opts);
/// ```
pub fn between(
    arr: Expression,
    lower: Expression,
    upper: Expression,
    options: BetweenOptions,
) -> Expression {
    Between
        .try_new_expr(options, [arr, lower, upper])
        .vortex_expect("Failed to create Between expression")
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use super::between;
    use crate::exprs::get_item::get_item;
    use crate::exprs::literal::lit;
    use crate::exprs::root::root;

    #[test]
    fn test_display() {
        let expr = between(
            get_item("score", root()),
            lit(10),
            lit(50),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        );
        assert_eq!(expr.to_string(), "(10i32 <= $.score < 50i32)");

        let expr2 = between(
            root(),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_eq!(expr2.to_string(), "(0i32 < $ <= 100i32)");
    }
}
