// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;
use vortex_vector::Datum;

use crate::compute::between as between_compute;
use crate::compute::BetweenOptions;
use crate::expr::expression::Expression;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::operators::Operator;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::ArrayRef;

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
    type Options = BetweenOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.between")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BetweenOpts {
                lower_strict: instance.lower_strict.is_strict(),
                upper_strict: instance.upper_strict.is_strict(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::BetweenOpts::decode(metadata)?;
        Ok(BetweenOptions {
            lower_strict: if opts.lower_strict {
                crate::compute::StrictComparison::Strict
            } else {
                crate::compute::StrictComparison::NonStrict
            },
            upper_strict: if opts.upper_strict {
                crate::compute::StrictComparison::Strict
            } else {
                crate::compute::StrictComparison::NonStrict
            },
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(3)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("array"),
            1 => ChildName::from("lower"),
            2 => ChildName::from("upper"),
            _ => unreachable!("Invalid child index {} for Between expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
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
            expr.child(1),
            lower_op,
            expr.child(0),
            upper_op,
            expr.child(2)
        )
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let arr_dt = &arg_dtypes[0];
        let lower_dt = &arg_dtypes[1];
        let upper_dt = &arg_dtypes[2];

        if !arr_dt.eq_ignore_nullability(lower_dt) {
            vortex_bail!(
                "Array dtype {} does not match lower dtype {}",
                arr_dt,
                lower_dt
            );
        }
        if !arr_dt.eq_ignore_nullability(upper_dt) {
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

    fn evaluate(
        &self,
        options: &Self::Options,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let arr = expr.child(0).evaluate(scope)?;
        let lower = expr.child(1).evaluate(scope)?;
        let upper = expr.child(2).evaluate(scope)?;
        between_compute(&arr, &lower, &upper, options)
    }

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let [arr, lower, upper]: [Datum; _] = args
            .datums
            .try_into()
            .map_err(|_| vortex_err!("Expected 3 arguments for Between expression",))?;
        let [arr_dt, lower_dt, upper_dt]: [DType; _] = args
            .dtypes
            .try_into()
            .map_err(|_| vortex_err!("Expected 3 dtypes for Between expression",))?;

        let lower_bound = Binary
            .bind(options.lower_strict.to_operator().into())
            .execute(ExecutionArgs {
                datums: vec![lower.clone(), arr.clone()],
                dtypes: vec![lower_dt.clone(), arr_dt.clone()],
                row_count: args.row_count,
                return_dtype: args.return_dtype.clone(),
            })?;
        let upper_bound = Binary
            .bind(options.upper_strict.to_operator().into())
            .execute(ExecutionArgs {
                datums: vec![arr.clone(), upper.clone()],
                dtypes: vec![arr_dt.clone(), upper_dt.clone()],
                row_count: args.row_count,
                return_dtype: args.return_dtype.clone(),
            })?;

        Binary.bind(Operator::And).execute(ExecutionArgs {
            datums: vec![lower_bound, upper_bound],
            dtypes: vec![args.return_dtype.clone(), args.return_dtype.clone()],
            row_count: args.row_count,
            return_dtype: args.return_dtype.clone(),
        })
    }

    fn stat_falsification(
        &self,
        options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let arr = expr.child(0).clone();
        let lower = expr.child(1).clone();
        let upper = expr.child(2).clone();

        let lhs = Binary.new_expr(
            options.lower_strict.to_operator().into(),
            [lower, arr.clone()],
        );
        let rhs = Binary.new_expr(options.upper_strict.to_operator().into(), [arr, upper]);

        Binary
            .new_expr(Operator::And, [lhs, rhs])
            .stat_falsification(catalog)
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
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
/// # use vortex_array::expr::{between, lit, root};
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
    use super::between;
    use crate::compute::BetweenOptions;
    use crate::compute::StrictComparison;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;

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
