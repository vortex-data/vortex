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
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DType::Bool;
use crate::expr::display::ExprDisplay;
use crate::expr::expression::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
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

/// Short-circuits between for the inputs that need no encoding-specific work.
///
/// Returns `Some(result)` when the answer is already known (empty array, null bounds), or `None`
/// when between must proceed with the encoding-specific implementation. Kernels can therefore rely
/// on both bounds being non-null.
///
/// The result can be a lazy [`ScalarFn`] array, so a caller that needs a computed array
/// **must** execute it.
///
/// [`ScalarFn`]: crate::arrays::ScalarFn
pub(super) fn short_circuit(
    arr: &ArrayRef,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
) -> VortexResult<Option<ArrayRef>> {
    let return_dtype =
        Bool(arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability());

    // Bail early if the array is empty.
    if arr.is_empty() {
        return Ok(Some(Canonical::empty(&return_dtype).into_array()));
    }

    let lower_is_null = lower.as_constant().is_some_and(|v| v.is_null());
    let upper_is_null = upper.as_constant().is_some_and(|v| v.is_null());

    // `Between` is not strict, and Kleene `AND` gives `null AND false = false`, so a null bound
    // cannot falsify a row on its own. Every row is null only when both bounds are null.
    if lower_is_null && upper_is_null {
        return Ok(Some(
            ConstantArray::new(Scalar::null(return_dtype), arr.len()).into_array(),
        ));
    }

    // Every kernel requires non-null constant bounds, so a single null bound leaves nothing to
    // dispatch to. The two compares keep the surviving bound, which can still falsify rows.
    if lower_is_null || upper_is_null {
        return as_two_compares(arr, lower, upper, options).map(Some);
    }

    Ok(None)
}

/// The two compares that `Between` stands for, combined with Kleene `AND`.
///
/// The returned array is lazy, so a reduce rule can call this function.
fn as_two_compares(
    arr: &ArrayRef,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    let lower_cmp = lower.binary(arr.clone(), options.lower_strict.to_operator())?;
    let upper_cmp = arr.binary(upper.clone(), options.upper_strict.to_operator())?;
    lower_cmp.binary(upper_cmp, Operator::And)
}

/// Between on a canonical array by directly dispatching to the appropriate kernel.
///
/// Falls back to [`as_two_compares`] if no kernel handles the input.
fn between_canonical(
    arr: &ArrayRef,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = short_circuit(arr, lower, upper, options)? {
        // TODO(joe): return the lazy array directly, blocked on the same executor support as the
        // fallback below. Only the single-null-bound case is lazy, so this forces it for now.
        return result.execute::<ArrayRef>(ctx);
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
    as_two_compares(arr, lower, upper, options)?.execute::<ArrayRef>(ctx)
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

impl Between {
    /// Creates a lazy between operation over an array and its bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(
        array: ArrayRef,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Between.bind(options), vec![array, lower, upper])
    }
}

impl ScalarFnVTable for Between {
    type Options = BetweenOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.between");
        *ID
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
        expr: &dyn ExprDisplay,
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
            expr.display_child(1),
            lower_op,
            expr.display_child(0),
            upper_op,
            expr.display_child(2)
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

    fn validity(
        &self,
        _options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // `Between` stands for two compares under Kleene `AND`, and `null AND false` is `false`,
        // so a null bound does not make a row null. There is no validity expression to derive,
        // which is also why `Binary` returns `None` for `Operator::And`.
        Ok(None)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        // Not strict for the same reason `validity` returns `None` above: under Kleene `AND` a
        // null bound does not force a null row.
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::between;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    const NON_STRICT: BetweenOptions = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };

    /// `len` null `i32` values held as a [`ConstantArray`], which is what `as_constant` sees.
    fn null_i32s(len: usize) -> ArrayRef {
        let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        ConstantArray::new(null, len).into_array()
    }

    /// A declared validity expression must agree with the mask of the executed result.
    ///
    /// The bounds are columns rather than literals so that a null bound reaches execution
    /// instead of being intercepted as a constant.
    #[test]
    fn validity_agrees_with_execution() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();

        //  x     lo     hi     expected
        //  10    null   5      false, since the upper bound alone falsifies the row
        //  10    null   50     null, since neither bound falsifies the row
        //  1     0      5      true
        let x = PrimitiveArray::from_option_iter([Some(10), Some(10), Some(1)]).into_array();
        let lo = PrimitiveArray::from_option_iter([None, None, Some(0)]).into_array();
        let hi = PrimitiveArray::from_option_iter([Some(5), Some(50), Some(5)]).into_array();
        let data = StructArray::from_fields(&[("x", x), ("lo", lo), ("hi", hi)])?.into_array();

        let expr = between(col("x"), col("lo"), col("hi"), NON_STRICT);

        let executed = data
            .clone()
            .apply(&expr)?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);

        let declared = data
            .apply(&expr.validity()?)?
            .execute::<BoolArray>(ctx)?
            .bool_vec(ctx);

        assert_eq!(executed, [Some(false), None, Some(true)]);
        assert_eq!(
            executed.iter().map(Option::is_some).collect::<Vec<_>>(),
            declared
        );

        Ok(())
    }

    #[test]
    fn is_not_strict() {
        let expr = between(
            root(),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );

        assert!(!expr.as_scalar().is_some_and(|f| f.signature().is_strict()));
    }

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
        let ctx = &mut SESSION.create_execution_ctx();

        let matches = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict,
                upper_strict,
            },
            ctx,
        )
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap();

        let indices = to_int_indices(matches, ctx).unwrap();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_constants() {
        let lower = buffer![0, 0, 2, 0, 2].into_array();
        let array = buffer![1, 0, 1, 0, 1].into_array();
        let ctx = &mut SESSION.create_execution_ctx();

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
            ctx,
        )
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap();

        // The rows the lower bound already falsified stay false rather than becoming null.
        assert_eq!(
            matches.opt_bool_vec(ctx),
            [None, None, Some(false), None, Some(false)]
        );

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
            ctx,
        )
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap();
        let indices = to_int_indices(matches, ctx).unwrap();
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
            ctx,
        )
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap();
        let indices = to_int_indices(matches, ctx).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    /// `Between` is not strict, so a null bound only makes a row null when the surviving
    /// comparison is not already false. This must not depend on how the bound is encoded, and
    /// compression stores an all-null chunk as a [`ConstantArray`].
    #[rstest]
    #[case::primitive_nulls(PrimitiveArray::from_option_iter([None::<i32>, None]).into_array())]
    #[case::constant_null(null_i32s(2))]
    fn null_lower_bound(#[case] lower: ArrayRef) -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let array = buffer![10, 10].into_array();
        let upper = buffer![5, 50].into_array();

        let result = between_canonical(&array, &lower, &upper, &NON_STRICT, ctx)?
            .execute::<BoolArray>(ctx)?;

        // Row 0 stays false because the upper bound falsifies it on its own.
        assert_eq!(result.opt_bool_vec(ctx), [Some(false), None]);

        Ok(())
    }

    /// With both bounds null no comparison can falsify a row, so every row is null.
    #[test]
    fn both_bounds_null() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let array = buffer![10, 10].into_array();
        let bound = null_i32s(2);

        let result = between_canonical(&array, &bound, &bound, &NON_STRICT, ctx)?
            .execute::<BoolArray>(ctx)?;

        assert_eq!(result.opt_bool_vec(ctx), [None, None]);

        Ok(())
    }

    #[test]
    fn test_between_decimal() {
        let ctx = &mut SESSION.create_execution_ctx();
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
            ctx,
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([false, true, true, true]),
            ctx
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
            ctx,
        )
        .unwrap();
        assert_arrays_eq!(
            between_strict,
            BoolArray::from_iter([true, true, true, false]),
            ctx
        );
    }

    /// Regression test for a fuzzer crash where a bound scalar used a wider storage type (I32)
    /// than the array's storage type (I16), causing the cast in `between_unpack` to fail.
    ///
    /// The fix casts the bound to the array's storage type and, when the cast fails, uses the
    /// overflow direction to determine the result without falling back to Arrow.
    #[rstest]
    // Upper bound too large (I32 > i16::MAX): upper constraint always satisfied → result from lower only.
    #[case(DecimalValue::I16(1), DecimalValue::I32(82246), vec![0, 1, 2, 3])]
    // Lower bound too large (I32 > i16::MAX): lower constraint never satisfied → all false.
    #[case(DecimalValue::I32(82246), DecimalValue::I16(4), vec![])]
    // Upper bound too small (negative I32 < i16::MIN): upper constraint never satisfied → all false.
    #[case(DecimalValue::I16(1), DecimalValue::I32(-82246), vec![])]
    // Lower bound too small (negative I32 < i16::MIN): lower constraint always satisfied → result from upper only.
    #[case(DecimalValue::I32(-82246), DecimalValue::I16(2), vec![0, 1])]
    fn test_between_decimal_mismatched_storage_types(
        #[case] lower_val: DecimalValue,
        #[case] upper_val: DecimalValue,
        #[case] expected_indices: Vec<u64>,
    ) {
        let ctx = &mut SESSION.create_execution_ctx();
        // Array uses I16 storage with precision=5 (values fit in i16 even though precision=5
        // nominally maps to I32 as the smallest storage type).
        let decimal_type = DecimalDType::new(5, -67);
        let array = DecimalArray::new(
            buffer![1i16, 2i16, 3i16, 4i16],
            decimal_type,
            Validity::NonNullable,
        )
        .into_array();

        let lower = ConstantArray::new(
            Scalar::decimal(lower_val, decimal_type, Nullability::NonNullable),
            array.len(),
        )
        .into_array();
        let upper = ConstantArray::new(
            Scalar::decimal(upper_val, decimal_type, Nullability::NonNullable),
            array.len(),
        )
        .into_array();

        let result = between_canonical(
            &array,
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            ctx,
        )
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap();

        assert_eq!(to_int_indices(result, ctx).unwrap(), expected_indices);
    }
}
