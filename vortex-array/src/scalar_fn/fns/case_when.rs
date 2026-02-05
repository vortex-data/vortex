// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary CASE WHEN expression for conditional value selection.
//!
//! This expression is a simple wrapper around the `zip` compute function:
//! `CASE WHEN condition THEN value ELSE else_value END`
//!
//! For n-ary CASE WHEN expressions (multiple WHEN clauses), use the
//! [`nested_case_when`] convenience function which converts to nested binary expressions:
//! `CASE WHEN a THEN x WHEN b THEN y ELSE z END` becomes
//! `CASE WHEN a THEN x ELSE (CASE WHEN b THEN y ELSE z END) END`

use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;

use prost::Message;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::compute::zip;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;

/// Options for the binary CaseWhen expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaseWhenOptions {
    /// Whether an ELSE clause is present.
    /// If false, unmatched rows return NULL.
    pub has_else: bool,
}

impl fmt::Display for CaseWhenOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "case_when(else={})", self.has_else)
    }
}

/// A binary CASE WHEN expression.
///
/// This is a simple conditional select: `CASE WHEN cond THEN value ELSE else_value END`
/// which is equivalent to `zip(value, else_value, cond)`.
///
/// Children are always in order: [condition, then_value, else_value?]
pub struct CaseWhen;

impl VTable for CaseWhen {
    type Options = CaseWhenOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.case_when")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let num_children = 2 + if options.has_else { 1 } else { 0 };
        Ok(Some(
            pb::CaseWhenOpts { num_children }.encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::CaseWhenOpts::decode(metadata)?;
        // We only support binary form:
        // - 2 children: [when, then]
        // - 3 children: [when, then, else]
        if !matches!(opts.num_children, 2 | 3) {
            vortex_bail!(
                "CaseWhen only supports binary form (2 or 3 children), got {}",
                opts.num_children
            );
        }
        Ok(CaseWhenOptions {
            has_else: opts.num_children == 3,
        })
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        // Binary: condition + then + optional else
        let num_children = 2 + if options.has_else { 1 } else { 0 };
        Arity::Exact(num_children)
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("when"),
            1 => ChildName::from("then"),
            2 if options.has_else => ChildName::from("else"),
            _ => unreachable!("Invalid child index {} for binary CaseWhen", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "CASE WHEN {} THEN {}", expr.child(0), expr.child(1))?;
        if options.has_else {
            write!(f, " ELSE {}", expr.child(2))?;
        }
        write!(f, " END")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        // The return dtype is based on the THEN expression (index 1)
        let then_dtype = &arg_dtypes[1];

        // If there's no ELSE, the result is always nullable (unmatched rows are NULL)
        if !options.has_else {
            Ok(then_dtype.as_nullable())
        } else {
            Ok(then_dtype.clone())
        }
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        let row_count = args.row_count;

        // Extract inputs based on arity: [condition, then_value] or [condition, then_value, else_value]
        let (condition, then_value, else_value) = match args.inputs.len() {
            2 => {
                let [condition, then_value]: [ArrayRef; 2] = args
                    .inputs
                    .try_into()
                    .map_err(|_| vortex_error::vortex_err!("Expected 2 inputs"))?;
                (condition, then_value, None)
            }
            3 => {
                let [condition, then_value, else_value]: [ArrayRef; 3] = args
                    .inputs
                    .try_into()
                    .map_err(|_| vortex_error::vortex_err!("Expected 3 inputs"))?;
                (condition, then_value, Some(else_value))
            }
            n => vortex_bail!("CaseWhen expects 2 or 3 inputs, got {}", n),
        };

        // Execute condition to get a BoolArray
        let cond_bool = condition.execute::<BoolArray>(args.ctx)?;
        // SQL semantics: NULL condition is treated as FALSE (i.e., we take the ELSE branch)
        let mask = cond_bool.to_mask_fill_null_false();

        // Short-circuit: all true -> just return THEN value
        if mask.all_true() {
            return then_value.execute::<ExecutionResult>(args.ctx);
        }

        // Short-circuit: all false -> return ELSE value or NULL
        if mask.all_false() {
            return match else_value {
                Some(else_value) => else_value.execute::<ExecutionResult>(args.ctx),
                None => {
                    // Create NULL constant of appropriate type
                    let then_dtype = then_value.dtype().as_nullable();
                    Ok(ExecutionResult::constant(
                        Scalar::null(then_dtype),
                        row_count,
                    ))
                }
            };
        }

        // Get else value for zip (create NULL constant if no else clause)
        let else_value = else_value.unwrap_or_else(|| {
            let then_dtype = then_value.dtype().as_nullable();
            ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
        });

        // Use zip to select: where mask is true, take then_value; else take else_value
        let result = zip(then_value.as_ref(), else_value.as_ref(), &mask)?;

        result.execute::<ExecutionResult>(args.ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        // CaseWhen is null-sensitive because NULL conditions are treated as false
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Creates a binary CASE WHEN expression with an ELSE clause.
///
/// # Arguments
/// - `condition`: Boolean expression for the WHEN clause
/// - `then_value`: Value to return when condition is true
/// - `else_value`: Value to return when condition is false
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END
/// case_when(gt(col("x"), lit(0)), lit("positive"), lit("non-positive"))
/// ```
pub fn case_when(
    condition: Expression,
    then_value: Expression,
    else_value: Expression,
) -> Expression {
    let options = CaseWhenOptions { has_else: true };
    CaseWhen.new_expr(options, [condition, then_value, else_value])
}

/// Creates a binary CASE WHEN expression without an ELSE clause.
///
/// Returns NULL when the condition is false.
///
/// # Arguments
/// - `condition`: Boolean expression for the WHEN clause
/// - `then_value`: Value to return when condition is true
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 0 THEN 'positive' END
/// case_when_no_else(gt(col("x"), lit(0)), lit("positive"))
/// ```
pub fn case_when_no_else(condition: Expression, then_value: Expression) -> Expression {
    let options = CaseWhenOptions { has_else: false };
    CaseWhen.new_expr(options, [condition, then_value])
}

/// Creates a nested CASE WHEN expression from multiple WHEN/THEN pairs.
///
/// This is a convenience function that converts n-ary CASE WHEN to nested binary expressions:
/// `CASE WHEN a THEN x WHEN b THEN y ELSE z END` becomes
/// `CASE WHEN a THEN x ELSE (CASE WHEN b THEN y ELSE z END) END`
///
/// # Arguments
/// - `when_then_pairs`: Vec of (condition, value) pairs
/// - `else_value`: Optional else expression (if None, unmatched rows return NULL)
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 10 THEN 'high' WHEN x > 5 THEN 'medium' ELSE 'low' END
/// nested_case_when(
///     vec![
///         (gt(col("x"), lit(10)), lit("high")),
///         (gt(col("x"), lit(5)), lit("medium")),
///     ],
///     Some(lit("low")),
/// )
/// ```
pub fn nested_case_when(
    when_then_pairs: Vec<(Expression, Expression)>,
    else_value: Option<Expression>,
) -> Expression {
    assert!(
        !when_then_pairs.is_empty(),
        "nested_case_when requires at least one when/then pair"
    );

    // Build from right to left (innermost first) using rfold
    when_then_pairs
        .into_iter()
        .rfold(else_value, |acc, (condition, then_value)| {
            Some(match acc {
                Some(else_expr) => case_when(condition, then_value, else_expr),
                None => case_when_no_else(condition, then_value),
            })
        })
        .unwrap_or_else(|| vortex_panic!("rfold on non-empty iterator always produces Some"))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect as _;
    use vortex_scalar::Scalar;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::VortexSessionExecute as _;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;
    use crate::expr::test_harness;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Helper to evaluate an expression using the apply+execute pattern
    fn evaluate_expr(expr: &Expression, array: &ArrayRef) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .apply(expr)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array()
    }

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialization_roundtrip() {
        let options = CaseWhenOptions { has_else: true };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen.deserialize(&serialized).unwrap();
        assert_eq!(options, deserialized);
    }

    #[test]
    fn test_serialization_no_else() {
        let options = CaseWhenOptions { has_else: false };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen.deserialize(&serialized).unwrap();
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
        // Becomes nested: CASE WHEN x>10 THEN 'high' ELSE (CASE WHEN x>5 THEN 'medium' ELSE 'low' END) END
        let expr = nested_case_when(
            vec![
                (gt(col("x"), lit(10i32)), lit("high")),
                (gt(col("x"), lit(5i32)), lit("medium")),
            ],
            Some(lit("low")),
        );
        let display = format!("{}", expr);
        // Should contain nested CASE statements
        assert_eq!(display.matches("CASE").count(), 2);
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

    // ==================== Arity Tests ====================

    #[test]
    fn test_arity_with_else() {
        let options = CaseWhenOptions { has_else: true };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(3));
    }

    #[test]
    fn test_arity_without_else() {
        let options = CaseWhenOptions { has_else: false };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(2));
    }

    // ==================== Child Name Tests ====================

    #[test]
    fn test_child_names() {
        let options = CaseWhenOptions { has_else: true };
        assert_eq!(CaseWhen.child_name(&options, 0).to_string(), "when");
        assert_eq!(CaseWhen.child_name(&options, 1).to_string(), "then");
        assert_eq!(CaseWhen.child_name(&options, 2).to_string(), "else");
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
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_nary_multiple_conditions() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[10, 0, 30, 0, 0]);
    }

    #[test]
    fn test_evaluate_nary_first_match_wins() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_no_else_returns_null() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when_no_else(gt(get_item("value", root()), lit(3i32)), lit(100i32));

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());

        assert_eq!(
            result.scalar_at(0).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(1).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(2).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(3).unwrap(),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(4).unwrap(),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
    }

    #[test]
    fn test_evaluate_all_conditions_false() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(100i32)),
            lit(1i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_evaluate_all_conditions_true() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(0i32)),
            lit(100i32),
            lit(0i32),
        );

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 100, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_with_literal_condition() {
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = case_when(lit(true), lit(100i32), lit(0i32));
        let result = evaluate_expr(&expr, &test_array);

        if let Some(constant) = result.as_constant() {
            assert_eq!(constant, Scalar::from(100i32));
        } else {
            let prim = result.to_primitive();
            assert_eq!(prim.as_slice::<i32>(), &[100, 100, 100]);
        }
    }

    #[test]
    fn test_evaluate_with_bool_column_result() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when(
            gt(get_item("value", root()), lit(2i32)),
            lit(true),
            lit(false),
        );

        let result = evaluate_expr(&expr, &test_array).to_bool();
        assert_eq!(
            result.to_bit_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, true, true, true]
        );
    }

    #[test]
    fn test_evaluate_with_nullable_condition() {
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when(get_item("cond", root()), lit(100i32), lit(0i32));

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 0, 0, 0, 100]);
    }

    #[test]
    fn test_evaluate_with_nullable_result_values() {
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
        let prim = result.to_primitive();
        assert_eq!(prim.as_slice::<i32>(), &[0, 0, 30, 40, 50]);
    }

    #[test]
    fn test_evaluate_with_all_null_condition() {
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([None, None, None]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when(get_item("cond", root()), lit(100i32), lit(0i32));

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0]);
    }

    // Note: Direct execute tests are covered through apply+execute tests above.

    // Note: The binary CASE WHEN implementation using `zip` does NOT provide
    // short-circuit/lazy evaluation. All child expressions are evaluated first,
    // then zip selects the result based on the condition. This means expressions
    // like divide-by-zero will still fail even if protected by a condition.
    // This is intentional - lazy evaluation would require a more complex
    // implementation that filters the input before evaluating children.
}
