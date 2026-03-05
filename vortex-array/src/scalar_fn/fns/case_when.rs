// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! N-ary CASE WHEN expression for conditional value selection.

use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
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
        ScalarFnId::from("vortex.case_when")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let num_children = options.num_when_then_pairs * 2 + u32::from(options.has_else);
        Ok(Some(pb::CaseWhenOpts { num_children }.encode_to_vec()))
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

        // The return dtype is based on the first THEN expression (index 1).
        // Validate all other THEN branches match and union their nullability.
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
            if !first_then.eq_ignore_nullability(else_dtype) {
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
        let row_count = args.row_count();
        let num_pairs = options.num_when_then_pairs as usize;

        let mut result: ArrayRef = if options.has_else {
            args.get(num_pairs * 2)?
        } else {
            let then_dtype = args.get(1)?.dtype().as_nullable();
            ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
        };

        for i in (0..num_pairs).rev() {
            let condition = args.get(i * 2)?;
            let then_value = args.get(i * 2 + 1)?;

            let cond_bool = condition.execute::<BoolArray>(ctx)?;
            let mask = cond_bool.to_mask_fill_null_false();

            if mask.all_true() {
                result = then_value;
                continue;
            }

            if mask.all_false() {
                continue;
            }

            result = zip_impl(&then_value, &result, &mask)?;
        }

        Ok(result)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        // CaseWhen is null-sensitive because NULL conditions are treated as false
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
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
    use crate::ToCanonical;
    use crate::VortexSessionExecute as _;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::case_when;
    use crate::expr::case_when_no_else;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::lit;
    use crate::expr::nested_case_when;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;
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

    // ==================== N-ary Evaluate Tests ====================

    #[test]
    fn test_evaluate_nary_no_else_returns_null() {
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

        assert_eq!(
            result.scalar_at(0).unwrap(),
            Scalar::from(10i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(1).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(2).unwrap(),
            Scalar::from(30i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(3).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(4).unwrap(),
            Scalar::null(result.dtype().clone())
        );
    }

    #[test]
    fn test_evaluate_nary_many_conditions() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_evaluate_nary_all_false_no_else() {
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
        for i in 0..3 {
            assert_eq!(
                result.scalar_at(i).unwrap(),
                Scalar::null(result.dtype().clone())
            );
        }
    }

    #[test]
    fn test_evaluate_nary_overlapping_conditions_first_wins() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // First matching condition always wins
        assert_eq!(result.as_slice::<i32>(), &[1, 1, 1]);
    }

    #[test]
    fn test_evaluate_nary_with_nullable_conditions() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // row 0: cond1=true → 10
        // row 1: cond1=NULL(→false), cond2=true → 20
        // row 2: cond1=false, cond2=NULL(→false) → else=0
        assert_eq!(result.as_slice::<i32>(), &[10, 20, 0]);
    }
}
