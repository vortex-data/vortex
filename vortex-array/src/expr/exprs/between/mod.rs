// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::any::Any;
use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalVTable;
use crate::arrays::PrimitiveVTable;
use crate::compute::BooleanOperator;
use crate::compute::Options;
use crate::compute::compare;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::execute_boolean;
use crate::expr::expression::Expression;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::operators::Operator;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BetweenOptions {
    pub lower_strict: StrictComparison,
    pub upper_strict: StrictComparison,
}

impl Display for BetweenOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let lower_op = if self.lower_strict.is_strict() {
            "<"
        } else {
            "<="
        };
        let upper_op = if self.upper_strict.is_strict() {
            "<"
        } else {
            "<="
        };
        write!(f, "lower_strict: {}, upper_strict: {}", lower_op, upper_op)
    }
}

impl Options for BetweenOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Strictness of the comparison.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StrictComparison {
    /// Strict bound (`<`)
    Strict,
    /// Non-strict bound (`<=`)
    NonStrict,
}

impl StrictComparison {
    pub const fn to_operator(&self) -> crate::compute::Operator {
        match self {
            StrictComparison::Strict => crate::compute::Operator::Lt,
            StrictComparison::NonStrict => crate::compute::Operator::Lte,
        }
    }

    pub const fn is_strict(&self) -> bool {
        matches!(self, StrictComparison::Strict)
    }
}

/// Common preconditions for between operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the between operation
/// (empty array, null bounds), or `None` if between should proceed with the
/// encoding-specific implementation.
pub(super) fn precondition(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
) -> VortexResult<Option<ArrayRef>> {
    let return_dtype =
        Bool(arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability());

    // Bail early if the array is empty.
    if arr.is_empty() {
        return Ok(Some(Canonical::empty(&return_dtype).into_array()));
    }

    // A quick check to see if either bound is a null constant array.
    if (lower.is_invalid(0)? || upper.is_invalid(0)?)
        && let (Some(c_lower), Some(c_upper)) = (lower.as_constant(), upper.as_constant())
        && (c_lower.is_null() || c_upper.is_null())
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(return_dtype), arr.len()).into_array(),
        ));
    }

    if lower.as_constant().is_some_and(|v| v.is_null())
        || upper.as_constant().is_some_and(|v| v.is_null())
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(return_dtype), arr.len()).into_array(),
        ));
    }

    Ok(None)
}

/// Between on a canonical array by directly dispatching to the appropriate kernel.
///
/// Falls back to compare + boolean and if no kernel handles the input.
pub(crate) fn between_canonical(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = precondition(arr, lower, upper)? {
        return Ok(result);
    }

    // Try type-specific kernels
    if let Some(prim) = arr.as_opt::<PrimitiveVTable>()
        && let Some(result) =
            <PrimitiveVTable as BetweenKernel>::between(prim, lower, upper, options, ctx)?
    {
        return Ok(result);
    }
    if let Some(dec) = arr.as_opt::<DecimalVTable>()
        && let Some(result) =
            <DecimalVTable as BetweenKernel>::between(dec, lower, upper, options, ctx)?
    {
        return Ok(result);
    }

    // TODO(joe): return lazy compare once the executor supports this
    // Fall back to compare + boolean and
    execute_boolean(
        &compare(lower, arr, options.lower_strict.to_operator())?,
        &compare(arr, upper, options.upper_strict.to_operator())?,
        BooleanOperator::AndKleene,
    )
}

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

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::BetweenOpts::decode(_metadata)?;
        Ok(BetweenOptions {
            lower_strict: if opts.lower_strict {
                StrictComparison::Strict
            } else {
                StrictComparison::NonStrict
            },
            upper_strict: if opts.upper_strict {
                StrictComparison::Strict
            } else {
                StrictComparison::NonStrict
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

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let [arr, lower, upper]: [ArrayRef; _] = args
            .inputs
            .try_into()
            .map_err(|_| vortex_err!("Expected 3 arguments for Between expression",))?;

        between_canonical(
            arr.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            options,
            args.ctx,
        )
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

    fn is_fallible(&self, _options: &Self::Options) -> bool {
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
    use super::*;
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
