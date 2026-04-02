// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::Primitive;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DType::Bool;
use crate::expr::StatsCatalog;
use crate::expr::expression::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::execute_boolean;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

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

/// Strictness of the comparison.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StrictComparison {
    /// Strict bound (`<`)
    Strict,
    /// Non-strict bound (`<=`)
    NonStrict,
}

impl StrictComparison {
    pub const fn to_compare_operator(&self) -> CompareOperator {
        match self {
            StrictComparison::Strict => CompareOperator::Lt,
            StrictComparison::NonStrict => CompareOperator::Lte,
        }
    }

    pub const fn to_operator(&self) -> Operator {
        match self {
            StrictComparison::Strict => Operator::Lt,
            StrictComparison::NonStrict => Operator::Lte,
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
    arr: &ArrayRef,
    lower: &ArrayRef,
    upper: &ArrayRef,
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
fn between_canonical(
    arr: &ArrayRef,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = precondition(arr, lower, upper)? {
        return Ok(result);
    }

    // Try type-specific kernels
    if let Some(prim) = arr.as_opt::<Primitive>()
        && let Some(result) =
            <Primitive as BetweenKernel>::between(prim, lower, upper, options, ctx)?
    {
        return Ok(result);
    }
    if let Some(dec) = arr.as_opt::<Decimal>()
        && let Some(result) = <Decimal as BetweenKernel>::between(dec, lower, upper, options, ctx)?
    {
        return Ok(result);
    }

    // TODO(joe): return lazy compare once the executor supports this
    // Fall back to compare + boolean and
    let lower_cmp = lower.clone().binary(
        arr.clone(),
        Operator::from(options.lower_strict.to_compare_operator()),
    )?;
    let upper_cmp = arr.clone().binary(
        upper.clone(),
        Operator::from(options.upper_strict.to_compare_operator()),
    )?;
    execute_boolean(&lower_cmp, &upper_cmp, Operator::And)
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
#[derive(Clone)]
pub struct Between;

impl ScalarFnVTable for Between {
    type Options = BetweenOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.between")
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

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let arr = args.get(0)?;
        let lower = args.get(1)?;
        let upper = args.get(2)?;

        // canonicalize the arr and we might be able to run a between kernels over that.
        if !arr.is_canonical() {
            return arr.execute::<Canonical>(ctx)?.into_array().between(
                lower,
                upper,
                options.clone(),
            );
        }

        between_canonical(&arr, &lower, &upper, options, ctx)
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

        let lhs = Binary.new_expr(options.lower_strict.to_operator(), [lower, arr.clone()]);
        let rhs = Binary.new_expr(options.upper_strict.to_operator(), [arr, upper]);

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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::ToCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::between;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

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

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict, vec![0, 1, 2, 3])]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict, vec![0, 1])]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict, vec![0, 2])]
    #[case(StrictComparison::Strict, StrictComparison::Strict, vec![0])]
    fn test_bounds(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
        #[case] expected: Vec<u64>,
    ) {
        let lower = buffer![0, 0, 0, 0, 2].into_array();
        let array = buffer![1, 0, 1, 0, 1].into_array();
        let upper = buffer![2, 1, 1, 0, 0].into_array();

        let matches = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict,
                upper_strict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .to_bool();

        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_constants() {
        let lower = buffer![0, 0, 2, 0, 2].into_array();
        let array = buffer![1, 0, 1, 0, 1].into_array();

        // upper is null
        let upper = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            5,
        )
        .into_array();

        let matches = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .to_bool();

        let indices = to_int_indices(matches).unwrap();
        assert!(indices.is_empty());

        // upper is a fixed constant
        let upper = ConstantArray::new(Scalar::from(2), 5).into_array();
        let matches = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .to_bool();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 3]);

        // lower is also a constant
        let lower = ConstantArray::new(Scalar::from(0), 5).into_array();

        let matches = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap()
        .to_bool();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_between_decimal() {
        let values = buffer![100i128, 200i128, 300i128, 400i128];
        let decimal_type = DecimalDType::new(3, 2);
        let array = DecimalArray::new(values, decimal_type, Validity::NonNullable).into_array();

        let lower = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(100i128),
                decimal_type,
                Nullability::NonNullable,
            ),
            array.len(),
        )
        .into_array();
        let upper = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(400i128),
                decimal_type,
                Nullability::NonNullable,
            ),
            array.len(),
        )
        .into_array();

        // Strict lower bound, non-strict upper bound
        let between_strict = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([false, true, true, true])
        );

        // Non-strict lower bound, strict upper bound
        let between_strict = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([true, true, true, false])
        );
    }
}
