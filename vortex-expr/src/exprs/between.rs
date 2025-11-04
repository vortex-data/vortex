// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Deref;
use vortex_array::compute::{between as between_compute, BetweenOptions};
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::v2::Expression;
use crate::{
    AnalysisExpr, Binary, ChildName, ExprId, ExprInstance, StatsCatalog, VTable, VTableExt,
};

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

    fn validate(&self, expr: &ExprInstance<Self>) -> VortexResult<()> {
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

    fn fmt_compact(&self, expr: &ExprInstance<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        let options = expr.metadata();
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

    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType> {
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

    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let arr = expr.child().evaluate(scope)?;
        let lower = expr.lower().evaluate(scope)?;
        let upper = expr.upper().evaluate(scope)?;
        between_compute(&arr, &lower, &upper, expr.deref())
    }
}

impl ExprInstance<'_, Between> {
    pub fn child(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn lower(&self) -> &Expression {
        &self.children()[1]
    }

    pub fn upper(&self) -> &Expression {
        &self.children()[2]
    }

    pub fn to_binary_expr(&self) -> VortexResult<Expression> {
        let options = self.metadata();
        let arr = self.children()[0].clone();
        let lower = self.children()[1].clone();
        let upper = self.children()[2].clone();

        let lhs = Binary.try_new(
            options.lower_strict.to_operator().into(),
            [lower, arr.clone()],
        )?;
        let rhs = Binary.try_new(options.upper_strict.to_operator().into(), [arr, upper])?;
        Binary.try_new(crate::Operator::And, [lhs, rhs])
    }
}

impl AnalysisExpr for Between {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        self.to_binary_expr().stat_falsification(catalog)
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
        .try_new(options, [arr.clone(), lower.clone(), upper.clone()])
        .vortex_expect("Failed to create Between expression")
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use crate::{between, get_item, lit, root};

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
