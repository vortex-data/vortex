// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQL-style CASE WHEN: evaluates `(condition, value)` pairs in order and returns
//! the value from the first matching condition (first-match-wins). NULL conditions
//! are treated as false. If no ELSE clause is provided, unmatched rows produce NULL;
//! otherwise they get the ELSE value.
//!
//! Unlike SQL which coerces all branches to a common supertype, all THEN/ELSE
//! branches must share the same base dtype (ignoring nullability). The result
//! nullability is the union of all branches (forced nullable if no ELSE).

use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::zip::zip_impl;

/// Options for the n-ary CaseWhen expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaseWhenOptions {
    /// Number of WHEN/THEN pairs.
    pub num_when_then_pairs: u32,
    /// Whether an ELSE clause is present.
    /// If false, unmatched rows return NULL.
    pub has_else: bool,
}

impl CaseWhenOptions {
    /// Total number of child expressions: 2 per WHEN/THEN pair, plus 1 if ELSE is present.
    pub fn num_children(&self) -> usize {
        self.num_when_then_pairs as usize * 2 + usize::from(self.has_else)
    }
}

impl fmt::Display for CaseWhenOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case_when(pairs={}, else={})",
            self.num_when_then_pairs, self.has_else
        )
    }
}

/// An n-ary CASE WHEN expression.
///
/// Children are in order: `[when_0, then_0, when_1, then_1, ..., else?]`.
#[derive(Clone)]
pub struct CaseWhen;

impl ScalarFnVTable for CaseWhen {
    type Options = CaseWhenOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.case_when");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        // let num_children = options.num_when_then_pairs * 2 + u32::from(options.has_else);
        // Ok(Some(pb::CaseWhenOpts { num_children }.encode_to_vec()))
        // stabilize the expr
        vortex_bail!("cannot serialize")
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::CaseWhenOpts::decode(metadata)?;
        if opts.num_children < 2 {
            vortex_bail!(
                "CaseWhen expects at least 2 children, got {}",
                opts.num_children
            );
        }
        Ok(CaseWhenOptions {
            num_when_then_pairs: opts.num_children / 2,
            has_else: opts.num_children % 2 == 1,
        })
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.num_children())
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        let num_pair_children = options.num_when_then_pairs as usize * 2;
        if child_idx < num_pair_children {
            let pair_idx = child_idx / 2;
            if child_idx.is_multiple_of(2) {
                ChildName::from(Arc::from(format!("when_{pair_idx}")))
            } else {
                ChildName::from(Arc::from(format!("then_{pair_idx}")))
            }
        } else if options.has_else && child_idx == num_pair_children {
            ChildName::from("else")
        } else {
            unreachable!("Invalid child index {} for CaseWhen", child_idx)
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "CASE")?;
        for i in 0..options.num_when_then_pairs as usize {
            write!(
                f,
                " WHEN {} THEN {}",
                expr.child(i * 2),
                expr.child(i * 2 + 1)
            )?;
        }
        if options.has_else {
            let else_idx = options.num_when_then_pairs as usize * 2;
            write!(f, " ELSE {}", expr.child(else_idx))?;
        }
        write!(f, " END")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if options.num_when_then_pairs == 0 {
            vortex_bail!("CaseWhen must have at least one WHEN/THEN pair");
        }

        let expected_len = options.num_children();
        if arg_dtypes.len() != expected_len {
            vortex_bail!(
                "CaseWhen expects {expected_len} argument dtypes, got {}",
                arg_dtypes.len()
            );
        }

        // Unlike SQL which coerces all branches to a common supertype, we require
        // all THEN/ELSE branches to have the same base dtype (ignoring nullability).
        // The result nullability is the union of all branches.
        let first_then = &arg_dtypes[1];
        let mut result_dtype = first_then.clone();

        for i in 1..options.num_when_then_pairs as usize {
            let then_i = &arg_dtypes[i * 2 + 1];
            if !first_then.eq_ignore_nullability(then_i) {
                vortex_bail!(
                    "CaseWhen THEN dtypes must match (ignoring nullability), got {} and {}",
                    first_then,
                    then_i
                );
            }
            result_dtype = result_dtype.union_nullability(then_i.nullability());
        }

        if options.has_else {
            let else_dtype = &arg_dtypes[options.num_when_then_pairs as usize * 2];
            if !result_dtype.eq_ignore_nullability(else_dtype) {
                vortex_bail!(
                    "CaseWhen THEN and ELSE dtypes must match (ignoring nullability), got {} and {}",
                    first_then,
                    else_dtype
                );
            }
            result_dtype = result_dtype.union_nullability(else_dtype.nullability());
        } else {
            // No ELSE means unmatched rows are NULL
            result_dtype = result_dtype.as_nullable();
        }

        Ok(result_dtype)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Inspired by https://datafusion.apache.org/blog/2026/02/02/datafusion_case/
        //
        // TODO: shrink input to `remaining` rows between WHEN iterations (batch reduction).
        // TODO: project to only referenced columns before batch reduction (column projection).
        // TODO: evaluate THEN/ELSE on compact matching/non-matching rows and scatter-merge the results.
        // TODO: for constant WHEN/THEN values, compile to a hash table for a single-pass lookup.
        let row_count = args.row_count();
        let num_pairs = options.num_when_then_pairs as usize;

        let mut remaining = Mask::new_true(row_count);
        let mut branches: Vec<(Mask, ArrayRef)> = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            if remaining.all_false() {
                break;
            }

            let condition = args.get(i * 2)?;
            let cond_bool = condition.execute::<BoolArray>(ctx)?;
            let cond_mask = cond_bool.to_mask_fill_null_false(ctx);
            let effective_mask = &remaining & &cond_mask;

            if effective_mask.all_false() {
                continue;
            }

            let then_value = args.get(i * 2 + 1)?;
            remaining = remaining.bitand_not(&cond_mask);
            branches.push((effective_mask, then_value));
        }

        let else_value: ArrayRef = if options.has_else {
            args.get(num_pairs * 2)?
        } else {
            let then_dtype = args.get(1)?.dtype().as_nullable();
            ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
        };

        if branches.is_empty() {
            return Ok(else_value);
        }

        merge_case_branches(branches, else_value, ctx)
    }

    fn simplify(
        &self,
        options: &Self::Options,
        expr: &Expression,
        _ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        // Rewrite the COALESCE-shaped CASE WHEN into `fill_null`, which references `x`
        // once and lowers to a single fill kernel instead of a `zip`/merge that resolves
        // `x` twice (once for the `is_null` predicate, once for the value branch).
        //
        //   CASE WHEN is_null(x)     THEN c ELSE x END  ==>  fill_null(x, c)
        //   CASE WHEN is_not_null(x) THEN x ELSE c END  ==>  fill_null(x, c)
        //
        // The fill `c` must be a `Literal`: `fill_null`'s kernel reads the fill value via
        // `as_constant()`, so a non-constant fill would produce an unexecutable expression.
        if options.num_when_then_pairs != 1 || !options.has_else {
            return Ok(None);
        }

        let when = expr.child(0);
        let then = expr.child(1);
        let els = expr.child(2);

        // `is_null(x) ? c : x` — predicate operand and ELSE are the same `x`, fill is THEN.
        let (x, fill) = if when.is::<IsNull>() && when.child(0) == els {
            (els, then)
        // `is_not_null(x) ? x : c` — predicate operand and THEN are the same `x`, fill is ELSE.
        } else if when.is::<IsNotNull>() && when.child(0) == then {
            (then, els)
        } else {
            return Ok(None);
        };

        let Some(scalar) = fill.as_opt::<Literal>() else {
            return Ok(None);
        };

        if scalar.is_null() {
            // Filling the nulls of `x` with NULL is a no-op
            return Ok(Some(x.clone()));
        }

        Ok(Some(crate::expr::fill_null(x.clone(), fill.clone())))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Average run length at which slicing + context-aware builder appends become cheaper than `scalar_at`.
/// Measured empirically via benchmarks.
const SLICE_CROSSOVER_RUN_LEN: usize = 4;

/// Merges disjoint `(mask, then_value)` branch pairs with an `else_value` into a single array.
///
/// Branch masks are guaranteed disjoint by the remaining-row tracking in [`CaseWhen::execute`].
fn merge_case_branches(
    branches: Vec<(Mask, ArrayRef)>,
    else_value: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if branches.len() == 1 {
        let (mask, then_value) = &branches[0];
        return zip_impl(then_value, &else_value, mask, ctx);
    }

    let output_nullability = branches
        .iter()
        .fold(else_value.dtype().nullability(), |acc, (_, arr)| {
            acc | arr.dtype().nullability()
        });
    let output_dtype = else_value.dtype().with_nullability(output_nullability);
    let branch_arrays: Vec<&ArrayRef> = branches.iter().map(|(_, arr)| arr).collect();

    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    for (branch_idx, (mask, _)) in branches.iter().enumerate() {
        match mask.slices() {
            AllOr::All => return branch_arrays[branch_idx].cast(output_dtype),
            AllOr::None => {}
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    spans.push((start, end, branch_idx));
                }
            }
        }
    }
    spans.sort_unstable_by_key(|&(start, ..)| start);

    if spans.is_empty() {
        return else_value.cast(output_dtype);
    }

    let builder = builder_with_capacity(&output_dtype, else_value.len());

    let fragmented = spans.len() > else_value.len() / SLICE_CROSSOVER_RUN_LEN;
    if fragmented {
        merge_row_by_row(
            &branch_arrays,
            &else_value,
            &spans,
            &output_dtype,
            builder,
            ctx,
        )
    } else {
        merge_run_by_run(
            &branch_arrays,
            &else_value,
            &spans,
            &output_dtype,
            builder,
            ctx,
        )
    }
}

/// Iterates spans directly, emitting one `scalar_at` per row.
/// Zero per-run allocations; preferred for fragmented masks (avg run < [`SLICE_CROSSOVER_RUN_LEN`]).
fn merge_row_by_row(
    branch_arrays: &[&ArrayRef],
    else_value: &ArrayRef,
    spans: &[(usize, usize, usize)],
    output_dtype: &DType,
    mut builder: Box<dyn ArrayBuilder>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut pos = 0;
    for &(start, end, branch_idx) in spans {
        for row in pos..start {
            let scalar = else_value.execute_scalar(row, ctx)?;
            builder.append_scalar(&scalar.cast(output_dtype)?)?;
        }
        for row in start..end {
            let scalar = branch_arrays[branch_idx].execute_scalar(row, ctx)?;
            builder.append_scalar(&scalar.cast(output_dtype)?)?;
        }
        pos = end;
    }
    for row in pos..else_value.len() {
        let scalar = else_value.execute_scalar(row, ctx)?;
        builder.append_scalar(&scalar.cast(output_dtype)?)?;
    }

    Ok(builder.finish())
}

/// Bulk-copies each span via `slice()` and context-aware builder appends.
/// Preferred when runs are long enough that memcpy dominates over per-slice allocation cost.
/// Lazy cast via `arr.cast(output_dtype)` is executed once per span as a block.
fn merge_run_by_run(
    branch_arrays: &[&ArrayRef],
    else_value: &ArrayRef,
    spans: &[(usize, usize, usize)],
    output_dtype: &DType,
    mut builder: Box<dyn ArrayBuilder>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let else_value = else_value.cast(output_dtype.clone())?;
    let len = else_value.len();
    for (start, end, branch_idx) in spans {
        if builder.len() < *start {
            else_value
                .slice(builder.len()..*start)?
                .append_to_builder(builder.as_mut(), ctx)?;
        }
        branch_arrays[*branch_idx]
            .cast(output_dtype.clone())?
            .slice(*start..*end)?
            .append_to_builder(builder.as_mut(), ctx)?;
    }
    if builder.len() < len {
        else_value
            .slice(builder.len()..len)?
            .append_to_builder(builder.as_mut(), ctx)?;
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::case_when;
    use crate::expr::case_when_no_else;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::is_not_null;
    use crate::expr::is_null;
    use crate::expr::lit;
    use crate::expr::nested_case_when;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    /// Helper to evaluate an expression using the apply+execute pattern
    fn evaluate_expr(expr: &Expression, array: &ArrayRef) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .clone()
            .apply(expr)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array()
    }

    // ==================== Serialization Tests ====================

    #[test]
    #[should_panic(expected = "cannot serialize")]
    fn test_serialization_roundtrip() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    #[test]
    #[should_panic(expected = "cannot serialize")]
    fn test_serialization_no_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: false,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display_with_else() {
        let expr = case_when(gt(col("value"), lit(0i32)), lit(100i32), lit(0i32));
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_no_else() {
        let expr = case_when_no_else(gt(col("value"), lit(0i32)), lit(100i32));
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(!display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_nested_nary() {
        // CASE WHEN x > 10 THEN 'high' WHEN x > 5 THEN 'medium' ELSE 'low' END
        let expr = nested_case_when(
            vec![
                (gt(col("x"), lit(10i32)), lit("high")),
                (gt(col("x"), lit(5i32)), lit("medium")),
            ],
            Some(lit("low")),
        );
        let display = format!("{}", expr);
        assert_eq!(display.matches("CASE").count(), 1);
        assert_eq!(display.matches("WHEN").count(), 2);
        assert_eq!(display.matches("THEN").count(), 2);
    }

    // ==================== DType Tests ====================

    #[test]
    fn test_return_dtype_with_else() {
        let expr = case_when(lit(true), lit(100i32), lit(0i32));
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_return_dtype_with_nullable_else() {
        let expr = case_when(
            lit(true),
            lit(100i32),
            lit(Scalar::null(DType::Primitive(
                PType::I32,
                Nullability::Nullable,
            ))),
        );
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_return_dtype_without_else_is_nullable() {
        let expr = case_when_no_else(lit(true), lit(100i32));
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_return_dtype_with_struct_input() {
        let dtype = test_harness::struct_dtype();
        let expr = case_when(
            gt(get_item("col1", root()), lit(10u16)),
            lit(100i32),
            lit(0i32),
        );
        let result_dtype = expr.return_dtype(&dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_return_dtype_mismatched_then_else_errors() {
        let expr = case_when(lit(true), lit(100i32), lit("zero"));
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let err = expr.return_dtype(&input_dtype).unwrap_err();
        assert!(
            err.to_string()
                .contains("THEN and ELSE dtypes must match (ignoring nullability)")
        );
    }

    // ==================== Arity Tests ====================

    #[test]
    fn test_arity_with_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(3));
    }

    #[test]
    fn test_arity_without_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: false,
        };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(2));
    }

    // ==================== Child Name Tests ====================

    #[test]
    fn test_child_names() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        assert_eq!(CaseWhen.child_name(&options, 0).to_string(), "when_0");
        assert_eq!(CaseWhen.child_name(&options, 1).to_string(), "then_0");
        assert_eq!(CaseWhen.child_name(&options, 2).to_string(), "else");
    }

    // ==================== N-ary Serialization Tests ====================

    #[test]
    #[should_panic(expected = "cannot serialize")]
    fn test_serialization_roundtrip_nary() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: true,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    #[test]
    #[should_panic(expected = "cannot serialize")]
    fn test_serialization_roundtrip_nary_no_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 4,
            has_else: false,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    // ==================== N-ary Arity Tests ====================

    #[test]
    fn test_arity_nary_with_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: true,
        };
        // 3 pairs * 2 children + 1 else = 7
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(7));
    }

    #[test]
    fn test_arity_nary_without_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: false,
        };
        // 3 pairs * 2 children = 6
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(6));
    }

    // ==================== N-ary Child Name Tests ====================

    #[test]
    fn test_child_names_nary() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: true,
        };
        assert_eq!(CaseWhen.child_name(&options, 0).to_string(), "when_0");
        assert_eq!(CaseWhen.child_name(&options, 1).to_string(), "then_0");
        assert_eq!(CaseWhen.child_name(&options, 2).to_string(), "when_1");
        assert_eq!(CaseWhen.child_name(&options, 3).to_string(), "then_1");
        assert_eq!(CaseWhen.child_name(&options, 4).to_string(), "when_2");
        assert_eq!(CaseWhen.child_name(&options, 5).to_string(), "then_2");
        assert_eq!(CaseWhen.child_name(&options, 6).to_string(), "else");
    }

    // ==================== N-ary DType Tests ====================

    #[test]
    fn test_return_dtype_nary_mismatched_then_types_errors() {
        let expr = nested_case_when(
            vec![(lit(true), lit(100i32)), (lit(false), lit("oops"))],
            Some(lit(0i32)),
        );
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let err = expr.return_dtype(&input_dtype).unwrap_err();
        assert!(err.to_string().contains("THEN dtypes must match"));
    }

    #[test]
    fn test_return_dtype_nary_mixed_nullability() {
        // When some THEN branches are nullable and others are not,
        // the result should be nullable (union of nullabilities).
        let non_null_then = lit(100i32);
        let nullable_then = lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        )));
        let expr = nested_case_when(
            vec![(lit(true), non_null_then), (lit(false), nullable_then)],
            Some(lit(0i32)),
        );
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(result, DType::Primitive(PType::I32, Nullability::Nullable));
    }

    #[test]
    fn test_return_dtype_nary_no_else_is_nullable() {
        let expr = nested_case_when(
            vec![(lit(true), lit(10i32)), (lit(false), lit(20i32))],
            None,
        );
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(result, DType::Primitive(PType::I32, Nullability::Nullable));
    }

    // ==================== Expression Manipulation Tests ====================

    #[test]
    fn test_replace_children() {
        let expr = case_when(lit(true), lit(1i32), lit(0i32));
        expr.with_children([lit(false), lit(2i32), lit(3i32)])
            .vortex_expect("operation should succeed in test");
    }

    // ==================== Evaluate Tests ====================

    #[test]
    fn test_evaluate_simple_condition() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            buffer![0i32, 0, 100, 100, 100].into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_nary_multiple_conditions() {
        let mut ctx = SESSION.create_execution_ctx();
        // Test n-ary via nested_case_when
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = nested_case_when(
            vec![
                (eq(get_item("value", root()), lit(1i32)), lit(10i32)),
                (eq(get_item("value", root()), lit(3i32)), lit(30i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![10i32, 0, 30, 0, 0].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_nary_first_match_wins() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // Both conditions match for values > 3, but first one wins
        let expr = nested_case_when(
            vec![
                (gt(get_item("value", root()), lit(2i32)), lit(100i32)),
                (gt(get_item("value", root()), lit(3i32)), lit(200i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            buffer![0i32, 0, 100, 100, 100].into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_no_else_returns_null() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when_no_else(gt(get_item("value", root()), lit(3i32)), lit(100i32));

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None::<i32>, None, None, Some(100), Some(100)])
                .into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_all_conditions_false() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(100i32)),
            lit(1i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![0i32, 0, 0, 0, 0].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_all_conditions_true() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(0i32)),
            lit(100i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            buffer![100i32, 100, 100, 100, 100].into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_all_true_no_else_returns_correct_dtype() {
        let mut ctx = SESSION.create_execution_ctx();
        // CASE WHEN value > 0 THEN 100 END — condition is always true, no ELSE.
        // Result must be Nullable because the implicit ELSE is NULL.
        let test_array = StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])
            .unwrap()
            .into_array();

        let expr = case_when_no_else(gt(get_item("value", root()), lit(0i32)), lit(100i32));

        let result = evaluate_expr(&expr, &test_array);
        assert!(
            result.dtype().is_nullable(),
            "result dtype must be Nullable, got {:?}",
            result.dtype()
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(100i32), Some(100), Some(100)]).into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_merge_case_branches_widens_nullability_of_later_branch() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // When a later THEN branch is Nullable and branches[0] and ELSE are NonNullable,
        // the result dtype must still be Nullable.
        //
        // CASE WHEN value = 0 THEN 10          -- NonNullable
        //      WHEN value = 1 THEN nullable(20) -- Nullable
        //      ELSE 0                           -- NonNullable
        // → result must be Nullable(i32)
        let test_array =
            StructArray::from_fields(&[("value", buffer![0i32, 1, 2].into_array())])?.into_array();

        let nullable_20 =
            Scalar::from(20i32).cast(&DType::Primitive(PType::I32, Nullability::Nullable))?;

        let expr = nested_case_when(
            vec![
                (eq(get_item("value", root()), lit(0i32)), lit(10i32)),
                (eq(get_item("value", root()), lit(1i32)), lit(nullable_20)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert!(
            result.dtype().is_nullable(),
            "result dtype must be Nullable, got {:?}",
            result.dtype()
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(20), Some(0)]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_evaluate_with_literal_condition() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = case_when(lit(true), lit(100i32), lit(0i32));
        let result = evaluate_expr(&expr, &test_array);

        assert_arrays_eq!(result, buffer![100i32, 100, 100].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_with_bool_column_result() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(2i32)),
            lit(true),
            lit(false),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, true, true, true]).into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_with_nullable_condition() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when(get_item("cond", root()), lit(100i32), lit(0i32));

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![100i32, 0, 0, 0, 100].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_with_nullable_result_values() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[
            ("value", buffer![1i32, 2, 3, 4, 5].into_array()),
            (
                "result",
                PrimitiveArray::from_option_iter([Some(10), None, Some(30), Some(40), Some(50)])
                    .into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(2i32)),
            get_item("result", root()),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0i32), Some(0), Some(30), Some(40), Some(50)])
                .into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_with_all_null_condition() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([None, None, None]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when(get_item("cond", root()), lit(100i32), lit(0i32));

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![0i32, 0, 0].into_array(), &mut ctx);
    }

    // ==================== N-ary Evaluate Tests ====================

    #[test]
    fn test_evaluate_nary_no_else_returns_null() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // Two conditions, no ELSE — unmatched rows should be NULL
        let expr = nested_case_when(
            vec![
                (eq(get_item("value", root()), lit(1i32)), lit(10i32)),
                (eq(get_item("value", root()), lit(3i32)), lit(30i32)),
            ],
            None,
        );

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), None, None])
                .into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_nary_many_conditions() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // 5 WHEN/THEN pairs: each value maps to its value * 10
        let expr = nested_case_when(
            vec![
                (eq(get_item("value", root()), lit(1i32)), lit(10i32)),
                (eq(get_item("value", root()), lit(2i32)), lit(20i32)),
                (eq(get_item("value", root()), lit(3i32)), lit(30i32)),
                (eq(get_item("value", root()), lit(4i32)), lit(40i32)),
                (eq(get_item("value", root()), lit(5i32)), lit(50i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(
            result,
            buffer![10i32, 20, 30, 40, 50].into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_nary_all_false_no_else() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])
            .unwrap()
            .into_array();

        // All conditions are false, no ELSE — everything should be NULL
        let expr = nested_case_when(
            vec![
                (gt(get_item("value", root()), lit(100i32)), lit(10i32)),
                (gt(get_item("value", root()), lit(200i32)), lit(20i32)),
            ],
            None,
        );

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array(),
            &mut ctx
        );
    }

    #[test]
    fn test_evaluate_nary_overlapping_conditions_first_wins() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            StructArray::from_fields(&[("value", buffer![10i32, 20, 30].into_array())])
                .unwrap()
                .into_array();

        // value=10: matches cond1 (>5) and cond2 (>0), first should win
        // value=20: matches all three, first should win
        // value=30: matches all three, first should win
        let expr = nested_case_when(
            vec![
                (gt(get_item("value", root()), lit(5i32)), lit(1i32)),
                (gt(get_item("value", root()), lit(0i32)), lit(2i32)),
                (gt(get_item("value", root()), lit(15i32)), lit(3i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        // First matching condition always wins
        assert_arrays_eq!(result, buffer![1i32, 1, 1].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_nary_early_exit_when_remaining_empty() {
        let mut ctx = SESSION.create_execution_ctx();
        // After branch 0 claims all rows, remaining becomes all_false.
        // The loop breaks before evaluating branch 1's condition.
        let test_array = StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])
            .unwrap()
            .into_array();

        let expr = nested_case_when(
            vec![
                (gt(get_item("value", root()), lit(0i32)), lit(100i32)),
                // Never evaluated due to early exit; 999 must never appear in output.
                (gt(get_item("value", root()), lit(0i32)), lit(999i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![100i32, 100, 100].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_nary_skips_branch_with_empty_effective_mask() {
        let mut ctx = SESSION.create_execution_ctx();
        // Branch 0 claims value=1. Branch 1 targets the same rows but they are already
        // matched → effective_mask is all_false → branch 1 is skipped (THEN not used).
        let test_array = StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])
            .unwrap()
            .into_array();

        let expr = nested_case_when(
            vec![
                (eq(get_item("value", root()), lit(1i32)), lit(10i32)),
                // Same condition as branch 0 — all matching rows already claimed → skipped.
                // 999 must never appear in output.
                (eq(get_item("value", root()), lit(1i32)), lit(999i32)),
                (eq(get_item("value", root()), lit(2i32)), lit(20i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_arrays_eq!(result, buffer![10i32, 20, 0].into_array(), &mut ctx);
    }

    #[test]
    fn test_evaluate_nary_string_output() -> VortexResult<()> {
        // Exercises merge_case_branches with a non-primitive (Utf8) builder.
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4].into_array())])?
                .into_array();

        // CASE WHEN value > 2 THEN 'high' WHEN value > 0 THEN 'low' ELSE 'none' END
        // value=1,2 → 'low' (branch 1 after branch 0 claims 3,4)
        // value=3,4 → 'high' (branch 0)
        let expr = nested_case_when(
            vec![
                (gt(get_item("value", root()), lit(2i32)), lit("high")),
                (gt(get_item("value", root()), lit(0i32)), lit("low")),
            ],
            Some(lit("none")),
        );

        let result = evaluate_expr(&expr, &test_array);
        assert_eq!(
            result.execute_scalar(0, &mut SESSION.create_execution_ctx())?,
            Scalar::utf8("low", Nullability::NonNullable)
        );
        assert_eq!(
            result.execute_scalar(1, &mut SESSION.create_execution_ctx())?,
            Scalar::utf8("low", Nullability::NonNullable)
        );
        assert_eq!(
            result.execute_scalar(2, &mut SESSION.create_execution_ctx())?,
            Scalar::utf8("high", Nullability::NonNullable)
        );
        assert_eq!(
            result.execute_scalar(3, &mut SESSION.create_execution_ctx())?,
            Scalar::utf8("high", Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn test_evaluate_nary_with_nullable_conditions() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[
            (
                "cond1",
                BoolArray::from_iter([Some(true), None, Some(false)]).into_array(),
            ),
            (
                "cond2",
                BoolArray::from_iter([Some(false), Some(true), None]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let expr = nested_case_when(
            vec![
                (get_item("cond1", root()), lit(10i32)),
                (get_item("cond2", root()), lit(20i32)),
            ],
            Some(lit(0i32)),
        );

        let result = evaluate_expr(&expr, &test_array);
        // row 0: cond1=true → 10
        // row 1: cond1=NULL(→false), cond2=true → 20
        // row 2: cond1=false, cond2=NULL(→false) → else=0
        assert_arrays_eq!(result, buffer![10i32, 20, 0].into_array(), &mut ctx);
    }

    // ==================== Simplify: COALESCE -> fill_null ====================

    /// Builds a non-nullable struct scope whose named fields are all `Nullable(I64)`.
    fn nullable_i64_scope(fields: &[&str]) -> DType {
        DType::Struct(
            StructFields::new(
                fields.to_vec().into(),
                vec![DType::Primitive(PType::I64, Nullability::Nullable); fields.len()],
            ),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn test_simplify_coalesce_is_null_rewrites_to_fill_null() -> VortexResult<()> {
        // CASE WHEN is_null(x) THEN 0 ELSE x END  ==>  fill_null(x, 0)
        let expr = case_when(is_null(col("x")), lit(0i64), col("x"));
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x"]))?;
        assert!(
            optimized.to_string().starts_with("vortex.fill_null"),
            "expected fill_null, got {optimized}"
        );
        Ok(())
    }

    #[test]
    fn test_simplify_coalesce_is_not_null_rewrites_to_fill_null() -> VortexResult<()> {
        // CASE WHEN is_not_null(x) THEN x ELSE 0 END  ==>  fill_null(x, 0)
        let expr = case_when(is_not_null(col("x")), col("x"), lit(0i64));
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x"]))?;
        assert!(
            optimized.to_string().starts_with("vortex.fill_null"),
            "expected fill_null, got {optimized}"
        );
        Ok(())
    }

    #[test]
    fn test_simplify_does_not_fire_when_operands_differ() -> VortexResult<()> {
        // The is_null operand (x) and the ELSE (y) are different columns: not a COALESCE.
        let expr = case_when(is_null(col("x")), lit(0i64), col("y"));
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x", "y"]))?;
        let s = optimized.to_string();
        assert!(s.contains("CASE"), "expected CASE WHEN to remain, got {s}");
        assert!(!s.contains("fill_null"), "must not rewrite, got {s}");
        Ok(())
    }

    #[test]
    fn test_simplify_does_not_fire_for_non_constant_fill() -> VortexResult<()> {
        // COALESCE(x, c) with a *column* fill: fill_null cannot consume a non-constant
        // fill value, so the rewrite must not fire.
        let expr = case_when(is_null(col("x")), col("c"), col("x"));
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x", "c"]))?;
        let s = optimized.to_string();
        assert!(s.contains("CASE"), "expected CASE WHEN to remain, got {s}");
        assert!(!s.contains("fill_null"), "must not rewrite, got {s}");
        Ok(())
    }

    #[test]
    fn test_simplify_null_fill_collapses_to_input() -> VortexResult<()> {
        // Filling the nulls of x with NULL is a no-op, so both forms collapse to just `x`.
        //   CASE WHEN is_null(x)     THEN null ELSE x    END  ==>  x
        //   CASE WHEN is_not_null(x) THEN x    ELSE null END  ==>  x
        let null_fill = || {
            lit(Scalar::null(DType::Primitive(
                PType::I64,
                Nullability::Nullable,
            )))
        };

        for expr in [
            case_when(is_null(col("x")), null_fill(), col("x")),
            case_when(is_not_null(col("x")), col("x"), null_fill()),
        ] {
            let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x"]))?;
            assert_eq!(
                optimized.to_string(),
                "$.x",
                "expected collapse to input column, got {optimized}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_simplify_null_fill_semantic_equivalence() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // The collapse-to-input rewrite must preserve values (and `x`'s nullability).
        let array = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
        let scope = DType::Primitive(PType::I64, Nullability::Nullable);
        let null_fill = lit(Scalar::null(DType::Primitive(
            PType::I64,
            Nullability::Nullable,
        )));

        let original = case_when(is_null(root()), null_fill, root());
        let optimized = original.optimize_recursive(&scope)?;
        assert_eq!(
            optimized.to_string(),
            "$",
            "expected collapse to root, got {optimized}"
        );

        let expected = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
        assert_arrays_eq!(evaluate_expr(&original, &array), expected, &mut ctx);
        assert_arrays_eq!(evaluate_expr(&optimized, &array), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_simplify_does_not_fire_without_else() -> VortexResult<()> {
        let expr = case_when_no_else(is_null(col("x")), lit(0i64));
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x"]))?;
        assert!(
            !optimized.to_string().contains("fill_null"),
            "must not rewrite a no-ELSE case_when, got {optimized}"
        );
        Ok(())
    }

    #[test]
    fn test_simplify_does_not_fire_for_multi_pair() -> VortexResult<()> {
        let expr = nested_case_when(
            vec![
                (is_null(col("x")), lit(0i64)),
                (gt(col("x"), lit(5i64)), lit(1i64)),
            ],
            Some(col("x")),
        );
        let optimized = expr.optimize_recursive(&nullable_i64_scope(&["x"]))?;
        assert!(
            !optimized.to_string().contains("fill_null"),
            "must not rewrite a multi-pair case_when, got {optimized}"
        );
        Ok(())
    }

    #[test]
    fn test_simplify_semantic_equivalence() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // The optimized expression must produce the same values as the original CASE WHEN.
        let array = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
        let scope = DType::Primitive(PType::I64, Nullability::Nullable);

        let original = case_when(is_null(root()), lit(0i64), root());
        let optimized = original.optimize_recursive(&scope)?;
        assert!(
            optimized.to_string().starts_with("vortex.fill_null"),
            "expected fill_null, got {optimized}"
        );

        // Original keeps CASE WHEN's nullable result dtype; the rewrite tightens it to
        // NonNullable because a non-null fill cannot leave any nulls behind. Values match.
        assert_arrays_eq!(
            evaluate_expr(&original, &array),
            PrimitiveArray::from_option_iter([Some(1i64), Some(0), Some(3)]).into_array(),
            &mut ctx
        );
        assert_arrays_eq!(
            evaluate_expr(&optimized, &array),
            buffer![1i64, 0, 3].into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_merge_case_branches_alternating_mask() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Exercises the scalar path: alternating rows produce one slice per row (no runs),
        // triggering the per-row cursor path in merge_case_branches.
        let n = 100usize;

        // Branch 0: even rows → 0, Branch 1: odd rows → 1, Else: never reached.
        let branch0_mask = Mask::from_indices(n, (0..n).step_by(2));
        let branch1_mask = Mask::from_indices(n, (1..n).step_by(2));

        let result = merge_case_branches(
            vec![
                (
                    branch0_mask,
                    PrimitiveArray::from_option_iter(vec![Some(0i32); n]).into_array(),
                ),
                (
                    branch1_mask,
                    PrimitiveArray::from_option_iter(vec![Some(1i32); n]).into_array(),
                ),
            ],
            PrimitiveArray::from_option_iter(vec![Some(99i32); n]).into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;

        // Even rows → 0, odd rows → 1.
        let expected: Vec<Option<i32>> = (0..n)
            .map(|v| if v % 2 == 0 { Some(0) } else { Some(1) })
            .collect();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter(expected).into_array(),
            &mut ctx
        );
        Ok(())
    }
}
