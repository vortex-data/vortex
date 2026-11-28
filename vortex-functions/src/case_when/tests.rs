// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::compute::conformance::hegel::draw_bool_pattern;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::NumericOperator;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_array::scalar_fn::builtin::fill_null::FillNull;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::col;
use vortex_expr::eq;
use vortex_expr::gt;
use vortex_expr::is_not_null;
use vortex_expr::lit;
use vortex_expr::proto::ExprSerializeProtoExt;
use vortex_expr::root;
use vortex_session::VortexSession;

use super::*;
use crate::expr::case_when;
use crate::expr::case_when_no_else;
use crate::expr::is_null;
use crate::expr::nested_case_when;
use crate::tests::draw_len;
use crate::tests::draw_nullability;
use crate::tests::slice_with_padding;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

#[derive(Clone, Copy, Debug)]
enum ValueKind {
    Bool,
    I32,
    F64,
    Decimal64,
    Decimal128,
    Utf8,
    Struct,
    Null,
}

const VALUE_KINDS: [ValueKind; 8] = [
    ValueKind::Bool,
    ValueKind::I32,
    ValueKind::F64,
    ValueKind::Decimal64,
    ValueKind::Decimal128,
    ValueKind::Utf8,
    ValueKind::Struct,
    ValueKind::Null,
];

const SPECIAL_F64S: [f64; 7] = [-0.0, 0.0, 1.5, -42.25, f64::NAN, -f64::NAN, f64::INFINITY];
const STR_POOL: [&str; 6] = [
    "",
    "a",
    "short",
    "twelve_bytes",
    "long out-of-line string value A",
    "long out-of-line string value B",
];

fn maybe_encode_array(
    tc: &TestCase,
    canonical: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = canonical.len();
    if matches!(canonical.dtype(), DType::Null) {
        return Ok(canonical);
    }

    match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 | 1 => Ok(canonical),
        2 => {
            let scalar = if len == 0 {
                Scalar::default_value(canonical.dtype())
            } else {
                canonical.execute_scalar(0, ctx)?
            };
            Ok(ConstantArray::new(scalar, len).into_array())
        }
        3 if len > 0 => {
            let dict_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(len.min(6)));
            let values = canonical.slice(0..dict_len)?;
            let max_code = u16::try_from(dict_len - 1).vortex_expect("fits u16");
            let codes: Vec<u16> = tc.draw(
                gs::vecs(gs::integers::<u16>().max_value(max_code))
                    .min_size(len)
                    .max_size(len),
            );
            let codes_validity = if canonical.dtype().is_nullable() {
                draw_validity_for(tc, Nullability::Nullable, len)
            } else {
                Validity::NonNullable
            };
            let codes_array = PrimitiveArray::new(Buffer::from(codes), codes_validity).into_array();
            Ok(DictArray::try_new(codes_array, values)?.into_array())
        }
        _ => slice_with_padding(tc, &canonical, ctx),
    }
}

fn draw_condition(
    tc: &TestCase,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<Option<bool>>)> {
    let nullability = draw_nullability(tc);
    let bits = draw_bool_pattern(tc, len);
    let validity = draw_validity_for(tc, nullability, len);
    let canonical = BoolArray::new(bits.into_iter().collect(), validity).into_array();
    let encoded = maybe_encode_array(tc, canonical, ctx)?;
    let truth = (0..len)
        .map(|row| Ok(encoded.execute_scalar(row, ctx)?.as_bool().value()))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok((encoded, truth))
}

fn draw_i32_field(tc: &TestCase, len: usize) -> ArrayRef {
    let nullability = draw_nullability(tc);
    let validity = draw_validity_for(tc, nullability, len);
    let vals: Vec<i32> = tc.draw(
        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50))
            .min_size(len)
            .max_size(len),
    );
    PrimitiveArray::new(Buffer::from(vals), validity).into_array()
}

fn draw_value_branch(
    tc: &TestCase,
    kind: ValueKind,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<Scalar>)> {
    let nullability = draw_nullability(tc);
    let validity = draw_validity_for(tc, nullability, len);
    let canonical: ArrayRef = match kind {
        ValueKind::Bool => {
            BoolArray::new(draw_bool_pattern(tc, len).into_iter().collect(), validity).into_array()
        }
        ValueKind::I32 => {
            let vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50))
                    .min_size(len)
                    .max_size(len),
            );
            PrimitiveArray::new(Buffer::from(vals), validity).into_array()
        }
        ValueKind::F64 => {
            let vals: Vec<f64> = (0..len)
                .map(|_| tc.draw(gs::sampled_from(&SPECIAL_F64S)))
                .collect();
            PrimitiveArray::new(Buffer::from(vals), validity).into_array()
        }
        ValueKind::Decimal64 => {
            let vals: Vec<i64> = tc.draw(
                gs::vecs(gs::integers::<i64>().min_value(-10_000).max_value(10_000))
                    .min_size(len)
                    .max_size(len),
            );
            DecimalArray::new(Buffer::from(vals), DecimalDType::new(15, 2), validity, ctx)
                .into_array()
        }
        ValueKind::Decimal128 => {
            let vals: Vec<i128> = (0..len)
                .map(|_| {
                    let base = tc.draw(gs::integers::<i16>().min_value(-100).max_value(100));
                    if tc.draw(gs::booleans()) {
                        (1i128 << 100) + i128::from(base)
                    } else {
                        i128::from(base)
                    }
                })
                .collect();
            DecimalArray::new(Buffer::from(vals), DecimalDType::new(38, 0), validity, ctx)
                .into_array()
        }
        ValueKind::Utf8 => {
            let mask = validity.execute_mask(len, ctx)?;
            let mut builder =
                builder_with_capacity_in(&DType::Utf8(nullability), len, ctx.allocator());
            for row in 0..len {
                let s = tc.draw(gs::sampled_from(&STR_POOL));
                if mask.value(row) {
                    builder.append_scalar(&Scalar::utf8(s, nullability))?;
                } else {
                    builder.append_null();
                }
            }
            builder.finish()
        }
        ValueKind::Struct => {
            // Two fields with independent nullabilities exercise asymmetric nested unions.
            let field_a = draw_i32_field(tc, len);
            let field_b = draw_i32_field(tc, len);
            StructArray::try_new(["a", "b"].into(), [field_a, field_b], len, validity)?.into_array()
        }
        ValueKind::Null => NullArray::new(len).into_array(),
    };

    let encoded = maybe_encode_array(tc, canonical, ctx)?;
    let scalars = (0..len)
        .map(|row| encoded.execute_scalar(row, ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok((encoded, scalars))
}

fn check_case_when_differential(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = draw_len(tc, 48);
    let num_pairs = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let has_else = tc.draw(gs::booleans());
    let kind = tc.draw(gs::sampled_from(&VALUE_KINDS));

    let mut cond_arrays = Vec::with_capacity(num_pairs);
    let mut cond_truths = Vec::with_capacity(num_pairs);
    let mut then_arrays = Vec::with_capacity(num_pairs);
    let mut then_scalars = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let (cond_arr, truth) = draw_condition(tc, len, &mut ctx)?;
        let (then_arr, scalars) = draw_value_branch(tc, kind, len, &mut ctx)?;
        cond_arrays.push(cond_arr);
        cond_truths.push(truth);
        then_arrays.push(then_arr);
        then_scalars.push(scalars);
    }

    let else_branch = if has_else {
        Some(draw_value_branch(tc, kind, len, &mut ctx)?)
    } else {
        None
    };

    let mut expected_dtype = then_arrays[0].dtype().clone();
    for arr in &then_arrays[1..] {
        expected_dtype = zip_nullability_union(&expected_dtype, arr.dtype())
            .vortex_expect("same base dtype across THEN branches");
    }
    if let Some((else_arr, _)) = &else_branch {
        expected_dtype = zip_nullability_union(&expected_dtype, else_arr.dtype())
            .vortex_expect("same base dtype across THEN and ELSE");
    } else {
        expected_dtype = expected_dtype.as_nullable();
    }

    // Exact row-by-row SQL CASE WHEN 3-valued truth reference.
    let mut expected_builder = builder_with_capacity_in(&expected_dtype, len, ctx.allocator());
    for row in 0..len {
        let matched = (0..num_pairs).find(|&p| cond_truths[p][row] == Some(true));
        let scalar = match matched {
            Some(p) => then_scalars[p][row].cast(&expected_dtype)?,
            None => match &else_branch {
                Some((_, else_scalars)) => else_scalars[row].cast(&expected_dtype)?,
                None => Scalar::null(expected_dtype.clone()),
            },
        };
        expected_builder.append_scalar(&scalar)?;
    }
    let expected = expected_builder.finish();

    // 1. Direct ScalarFnArray evaluation.
    let mut children = Vec::with_capacity(num_pairs * 2 + usize::from(has_else));
    for i in 0..num_pairs {
        children.push(cond_arrays[i].clone());
        children.push(then_arrays[i].clone());
    }
    if let Some((else_arr, _)) = &else_branch {
        children.push(else_arr.clone());
    }
    let options = CaseWhenOptions {
        num_when_then_pairs: u32::try_from(num_pairs).vortex_expect("fits u32"),
        has_else,
    };
    let direct = ScalarFnArray::try_new(CaseWhen.bind(options), children)?.into_array();
    assert_eq!(direct.dtype(), &expected_dtype);
    assert_arrays_eq!(direct, expected, &mut ctx);

    // 2. Expression evaluation (unoptimized and recursively optimized).
    let mut fields: Vec<(String, ArrayRef)> = Vec::new();
    let mut expr_pairs = Vec::with_capacity(num_pairs);
    for i in 0..num_pairs {
        let c_name = format!("c{i}");
        let v_name = format!("v{i}");
        fields.push((c_name.clone(), cond_arrays[i].clone()));
        fields.push((v_name.clone(), then_arrays[i].clone()));
        expr_pairs.push((col(c_name), col(v_name)));
    }
    let expr_else = else_branch.as_ref().map(|(else_arr, _)| {
        fields.push(("else".to_owned(), else_arr.clone()));
        col("else")
    });
    let struct_input = StructArray::try_new(
        fields.iter().map(|(n, _)| n.as_str()).collect(),
        fields.into_iter().map(|(_, a)| a),
        len,
        Validity::NonNullable,
    )?
    .into_array();

    let expr = nested_case_when(expr_pairs, expr_else);
    assert_eq!(expr.return_dtype(struct_input.dtype())?, expected_dtype);

    let applied = struct_input.clone().apply(&expr)?;
    assert_eq!(applied.dtype(), &expected_dtype);
    assert_arrays_eq!(applied, expected, &mut ctx);

    let optimized = expr.bind(struct_input.dtype())?.optimize_recursive()?;
    let opt_applied = struct_input.apply_bound(&optimized)?;
    assert_eq!(opt_applied.dtype(), &expected_dtype);
    assert_arrays_eq!(opt_applied, expected, &mut ctx);

    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn hegel_case_when_differential_oracle(tc: TestCase) {
    check_case_when_differential(&tc).vortex_expect("case_when differential oracle failed");
}

fn check_coalesce_simplify_equivalence(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = draw_len(tc, 24);
    let is_null_form = tc.draw(gs::booleans());
    let null_fill = tc.draw(gs::booleans());

    let (input, fill_scalar): (ArrayRef, Scalar) = match tc.draw(gs::integers::<u8>().max_value(5))
    {
        0 => {
            let (arr, _) = draw_value_branch(tc, ValueKind::I32, len, &mut ctx)?;
            let fill = if null_fill {
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
            } else {
                Scalar::primitive(tc.draw(gs::integers::<i32>()), draw_nullability(tc))
            };
            (arr, fill)
        }
        1 => {
            let (arr, _) = draw_value_branch(tc, ValueKind::Utf8, len, &mut ctx)?;
            let fill = if null_fill {
                Scalar::null(DType::Utf8(Nullability::Nullable))
            } else {
                Scalar::utf8(tc.draw(gs::sampled_from(&STR_POOL)), draw_nullability(tc))
            };
            (arr, fill)
        }
        2 => {
            let (arr, _) = draw_value_branch(tc, ValueKind::Decimal64, len, &mut ctx)?;
            let dec = DecimalDType::new(15, 2);
            let fill = if null_fill {
                Scalar::null(DType::Decimal(dec, Nullability::Nullable))
            } else {
                Scalar::decimal(
                    tc.draw(gs::integers::<i64>().min_value(-999).max_value(999))
                        .into(),
                    dec,
                    draw_nullability(tc),
                )
            };
            (arr, fill)
        }
        3 => {
            let (arr, _) = draw_value_branch(tc, ValueKind::Struct, len, &mut ctx)?;
            let a_null = draw_nullability(tc);
            let b_null = draw_nullability(tc);
            let struct_dtype = DType::Struct(
                StructFields::new(
                    ["a", "b"].into(),
                    vec![
                        DType::Primitive(PType::I32, a_null),
                        DType::Primitive(PType::I32, b_null),
                    ],
                ),
                draw_nullability(tc),
            );
            let fill = if null_fill {
                Scalar::null(struct_dtype.as_nullable())
            } else {
                Scalar::struct_(
                    struct_dtype,
                    vec![
                        Scalar::primitive(7i32, a_null),
                        Scalar::primitive(11i32, b_null),
                    ],
                )
            };
            (arr, fill)
        }
        4 => {
            let input_elem_null = draw_nullability(tc);
            let fill_elem_null = draw_nullability(tc);
            let input_elem_dtype = Arc::new(DType::Primitive(PType::I32, input_elem_null));
            let input_dtype = DType::List(Arc::clone(&input_elem_dtype), draw_nullability(tc));
            let mask = draw_validity_for(tc, input_dtype.nullability(), len)
                .execute_mask(len, &mut ctx)?;
            let mut builder = builder_with_capacity_in(&input_dtype, len, ctx.allocator());
            for row in 0..len {
                if mask.value(row) {
                    let elem = Scalar::primitive(
                        tc.draw(gs::integers::<i32>().min_value(-5).max_value(5)),
                        input_elem_null,
                    );
                    builder.append_scalar(&Scalar::list(
                        Arc::clone(&input_elem_dtype),
                        vec![elem],
                        input_dtype.nullability(),
                    ))?;
                } else {
                    builder.append_null();
                }
            }
            let arr = maybe_encode_array(tc, builder.finish(), &mut ctx)?;
            let fill_elem_dtype = Arc::new(DType::Primitive(PType::I32, fill_elem_null));
            let fill = if null_fill {
                Scalar::null(DType::List(fill_elem_dtype, Nullability::Nullable))
            } else {
                let elem = if fill_elem_null.is_nullable() && tc.draw(gs::booleans()) {
                    Scalar::null(fill_elem_dtype.as_ref().clone())
                } else {
                    Scalar::primitive(99i32, fill_elem_null)
                };
                Scalar::list(fill_elem_dtype, vec![elem], draw_nullability(tc))
            };
            (arr, fill)
        }
        _ => (NullArray::new(len).into_array(), Scalar::null(DType::Null)),
    };

    let original = if is_null_form {
        case_when(is_null(root()), lit(fill_scalar.clone()), root())
    } else {
        case_when(is_not_null(root()), root(), lit(fill_scalar.clone()))
    };
    let original_dtype = original.return_dtype(input.dtype())?;
    let optimized = original.bind(input.dtype())?.optimize_recursive()?;

    let bound_root = root().bind(input.dtype())?;
    if fill_scalar.is_null() {
        assert_eq!(
            optimized, bound_root,
            "null fill must collapse to input, got {optimized}"
        );
    } else if FillNull::supports_dtype(input.dtype())
        && input.dtype().as_nullable() == original_dtype.as_nullable()
    {
        if input.dtype().is_nullable() {
            assert!(
                optimized.is::<FillNull>(),
                "nullable COALESCE must rewrite to fill_null, got {optimized}"
            );
        } else {
            assert!(
                optimized.is::<FillNull>() || optimized == bound_root,
                "non-nullable COALESCE must rewrite to fill_null or root, got {optimized}"
            );
        }
    } else {
        assert!(
            optimized.is::<CaseWhen>(),
            "unsupported or nested-nullability-widening COALESCE must stay CaseWhen, got {optimized}"
        );
    }

    let orig_result = input.clone().apply(&original)?;
    let opt_result = input.apply_bound(&optimized)?;
    if optimized.is::<FillNull>() && !fill_scalar.dtype().is_nullable() {
        assert!(!opt_result.dtype().is_nullable());
    }
    assert_arrays_eq!(opt_result.cast(original_dtype)?, orig_result, &mut ctx);
    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn hegel_case_when_coalesce_simplify_oracle(tc: TestCase) {
    check_coalesce_simplify_equivalence(&tc).vortex_expect("COALESCE simplify oracle failed");
}

/// Regression tests for nested nullability handling in `CaseWhen::simplify`:
/// - `list(i32?)?` + `list(i32)` fill widens the fill scalar and rewrites to `FillNull`.
/// - `struct{a: i32?, b: i32}?` + `struct{a: i32, b: i32}` fill widens field `a` and rewrites.
/// - Asymmetric `struct{a: i32?, b: i32}?` + `struct{a: i32, b: i32?}` fill stays `CaseWhen`.
#[test]
fn test_simplify_coalesce_nested_nullability_cases() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let i32_null = DType::Primitive(PType::I32, Nullability::Nullable);
    let i32_nn = DType::Primitive(PType::I32, Nullability::NonNullable);

    // 1. List with nullable elements filled by a literal with non-nullable elements.
    let list_dtype = DType::List(Arc::new(i32_null.clone()), Nullability::Nullable);
    let list_fill = Scalar::list(
        Arc::new(i32_nn.clone()),
        vec![Scalar::from(9i32)],
        Nullability::NonNullable,
    );
    let list_expr = case_when(is_not_null(root()), root(), lit(list_fill));
    let list_opt = list_expr.bind(&list_dtype)?.optimize_recursive()?;
    assert!(list_opt.is::<FillNull>(), "{list_opt}");

    let list_with_null_elem = Scalar::list(
        Arc::new(i32_null.clone()),
        vec![Scalar::null(i32_null.clone())],
        Nullability::Nullable,
    );
    let mut lb = builder_with_capacity_in(&list_dtype, 2, ctx.allocator());
    lb.append_scalar(&list_with_null_elem)?;
    lb.append_null();
    let mixed_list = lb.finish();
    // Deterministically exercise `short_circuit` (empty / all-valid) and `fill_null_constant`
    // (all-null and non-null ConstantArray) as well as mixed-validity canonical inputs.
    for input in [
        builder_with_capacity_in(&list_dtype, 0, ctx.allocator()).finish(),
        mixed_list,
        ConstantArray::new(Scalar::null(list_dtype.clone()), 3).into_array(),
        ConstantArray::new(list_with_null_elem, 3).into_array(),
    ] {
        assert_arrays_eq!(
            input
                .clone()
                .apply_bound(&list_opt)?
                .cast(list_dtype.clone())?,
            input.apply(&list_expr)?,
            &mut ctx
        );
    }

    // 2. Two-field struct: `struct{a: i32?, b: i32}?` + `struct{a: i32, b: i32}` rewrites.
    let x_struct_dtype = DType::Struct(
        StructFields::new(["a", "b"].into(), vec![i32_null.clone(), i32_nn.clone()]),
        Nullability::Nullable,
    );
    let nn_struct_dtype = DType::Struct(
        StructFields::new(["a", "b"].into(), vec![i32_nn.clone(), i32_nn.clone()]),
        Nullability::NonNullable,
    );
    let struct_fill = Scalar::struct_(
        nn_struct_dtype,
        vec![Scalar::from(1i32), Scalar::from(2i32)],
    );
    let struct_expr = case_when(is_null(root()), lit(struct_fill), root());
    let struct_opt = struct_expr.bind(&x_struct_dtype)?.optimize_recursive()?;
    assert!(struct_opt.is::<FillNull>(), "{struct_opt}");

    let struct_with_null_field = Scalar::struct_(
        x_struct_dtype.clone(),
        vec![Scalar::null(i32_null.clone()), Scalar::from(5i32)],
    );
    let mut sb = builder_with_capacity_in(&x_struct_dtype, 2, ctx.allocator());
    sb.append_scalar(&struct_with_null_field)?;
    sb.append_null();
    let mixed_struct = sb.finish();
    for input in [
        builder_with_capacity_in(&x_struct_dtype, 0, ctx.allocator()).finish(),
        mixed_struct.clone(),
        ConstantArray::new(Scalar::null(x_struct_dtype.clone()), 3).into_array(),
        ConstantArray::new(struct_with_null_field, 3).into_array(),
    ] {
        assert_arrays_eq!(
            input
                .clone()
                .apply_bound(&struct_opt)?
                .cast(x_struct_dtype.clone())?,
            input.apply(&struct_expr)?,
            &mut ctx
        );
    }

    // 3. Asymmetric opposite-direction nested nullability stays `CaseWhen`.
    let asym_fill_dtype = DType::Struct(
        StructFields::new(["a", "b"].into(), vec![i32_nn, i32_null.clone()]),
        Nullability::NonNullable,
    );
    let asym_fill = Scalar::struct_(
        asym_fill_dtype,
        vec![Scalar::from(1i32), Scalar::null(i32_null)],
    );
    let asym_expr = case_when(is_null(root()), lit(asym_fill), root());
    let asym_opt = asym_expr.bind(&x_struct_dtype)?.optimize_recursive()?;
    assert!(asym_opt.is::<CaseWhen>(), "{asym_opt}");
    assert_arrays_eq!(
        mixed_struct.clone().apply_bound(&asym_opt)?,
        mixed_struct.apply(&asym_expr)?,
        &mut ctx
    );
    Ok(())
}

// ==================== Lazy Short-Circuiting & Merge Threshold Tests ====================

#[test]
fn selected_rows_do_not_execute_erroring_branches_or_later_conditions() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let condition = BoolArray::from_iter([Some(true), Some(false), Some(true), None]).into_array();
    let division = buffer![20i32, 30, 40, 50]
        .into_array()
        .numeric(buffer![2i32, 0, 4, 0].into_array(), NumericOperator::Div)?;
    assert!(division.clone().execute::<Canonical>(&mut ctx).is_err());

    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: true,
    };
    let result = CaseWhen.execute(
        &options,
        &VecExecutionArgs::new(
            vec![
                condition.clone(),
                division.clone(),
                ConstantArray::new(7i32, 4).into_array(),
            ],
            4,
        ),
        &mut ctx,
    )?;
    assert_arrays_eq!(result, buffer![10i32, 7, 10, 7].into_array(), &mut ctx);

    let result = CaseWhen.execute(
        &options,
        &VecExecutionArgs::new(
            vec![
                condition.clone(),
                StructArray::try_from_iter([("value", division.clone())])?.into_array(),
                StructArray::try_from_iter([("value", ConstantArray::new(7i32, 4).into_array())])?
                    .into_array(),
            ],
            4,
        ),
        &mut ctx,
    )?;
    assert_arrays_eq!(
        result,
        StructArray::try_from_iter([("value", buffer![10i32, 7, 10, 7].into_array())])?
            .into_array(),
        &mut ctx
    );

    let later_condition = buffer![20i32, 30, 40, 50]
        .into_array()
        .numeric(buffer![0i32, 3, 0, 5].into_array(), NumericOperator::Div)?
        .compare(
            ConstantArray::new(0i32, 4).into_array(),
            CompareOperator::Gt,
        )?;
    let result = CaseWhen.execute(
        &CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        },
        &VecExecutionArgs::new(
            vec![
                condition,
                ConstantArray::new(2i32, 4).into_array(),
                later_condition,
                ConstantArray::new(3i32, 4).into_array(),
                division,
            ],
            4,
        ),
        &mut ctx,
    )?;
    assert_arrays_eq!(result, buffer![2i32, 3, 2, 3].into_array(), &mut ctx);

    let selected_error = ScalarFnArray::try_new(
        CaseWhen.bind(options),
        vec![
            ConstantArray::new(true, 4).into_array(),
            buffer![20i32, 30, 40, 50]
                .into_array()
                .numeric(buffer![2i32, 0, 4, 0].into_array(), NumericOperator::Div)?,
            ConstantArray::new(7i32, 4).into_array(),
        ],
    )?
    .into_array();
    assert!(selected_error.execute::<Canonical>(&mut ctx).is_err());
    Ok(())
}

/// Exercises both `merge_compact` (alternating 300-row mask) and `merge_runs` (coarse 300-row
/// mask) through public `CaseWhen` evaluation.
#[rstest]
#[case::alternating_compact(false)]
#[case::coarse_runs(true)]
fn case_when_exercises_coarse_and_fragmented_merges(#[case] coarse: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let n = 300usize;
    let cond0 = BoolArray::from_iter((0..n).map(|i| if coarse { i < 150 } else { i % 2 == 0 }))
        .into_array();
    let cond1 = BoolArray::from_iter((0..n).map(|i| if coarse { i >= 150 } else { i % 2 == 1 }))
        .into_array();
    let result = ScalarFnArray::try_new(
        CaseWhen.bind(CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        }),
        vec![
            cond0,
            ConstantArray::new(0i32, n).into_array(),
            cond1,
            ConstantArray::new(1i32, n).into_array(),
            ConstantArray::new(99i32, n).into_array(),
        ],
    )?
    .into_array();

    let expected: Vec<i32> = (0..n)
        .map(|i| {
            if (coarse && i < 150) || (!coarse && i % 2 == 0) {
                0
            } else {
                1
            }
        })
        .collect();
    assert_arrays_eq!(result, Buffer::from(expected).into_array(), &mut ctx);
    Ok(())
}

// ==================== Serde & Guard Tables ====================

#[rstest]
#[case(1, true, 3)]
#[case(1, false, 2)]
#[case(3, true, 7)]
#[case(4, false, 8)]
#[case(u32::MAX / 2, true, u32::MAX)]
#[case(u32::MAX / 2, false, u32::MAX - 1)]
fn test_serialization_roundtrip(
    #[case] num_when_then_pairs: u32,
    #[case] has_else: bool,
    #[case] num_children: u32,
) -> VortexResult<()> {
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    let serialized = CaseWhen
        .serialize(&options)?
        .vortex_expect("serializable options");
    assert_eq!(
        pb::CaseWhenOpts::decode(serialized.as_slice())?.num_children,
        num_children
    );
    assert_eq!(CaseWhen.deserialize(&serialized, &SESSION)?, options);
    Ok(())
}

#[rstest]
#[case(0, "at least one WHEN/THEN pair")]
#[case(u32::MAX / 2 + 1, "too many children")]
fn test_serialization_invalid_options(
    #[case] num_when_then_pairs: u32,
    #[case] expected_error: &str,
    #[values(false, true)] has_else: bool,
) {
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    let error = CaseWhen.serialize(&options).unwrap_err();
    assert!(error.to_string().contains(expected_error), "{error}");
}

#[rstest]
#[case::missing(&[])]
#[case::zero_children(&[0x08, 0])]
#[case::one_child(&[0x08, 1])]
#[case::truncated(&[0x08, 0x80])]
fn test_deserialization_invalid_metadata(#[case] metadata: &[u8]) {
    assert!(CaseWhen.deserialize(metadata, &SESSION).is_err());
}

#[rstest]
fn test_expression_serialization_roundtrip(
    #[values(1, 3)] num_when_then_pairs: u32,
    #[values(false, true)] has_else: bool,
) -> VortexResult<()> {
    let branches = (0..num_when_then_pairs)
        .map(|value| (eq(col("value"), lit(value)), lit(value)))
        .collect();
    let expression = nested_case_when(branches, has_else.then(|| lit(42u32)));
    let bytes = expression.serialize_proto()?.encode_to_vec();
    let proto = pb::Expr::decode(bytes.as_slice())?;
    assert_eq!(proto.id, "vortex.case_when");
    assert_eq!(Expression::from_proto(&proto, &SESSION)?, expression);
    Ok(())
}

#[rstest]
#[case::zero_pairs(
    CaseWhenOptions { num_when_then_pairs: 0, has_else: false },
    vec![],
    "at least one WHEN/THEN pair",
)]
#[case::wrong_arity(
    CaseWhenOptions { num_when_then_pairs: 1, has_else: true },
    vec![DType::Bool(Nullability::NonNullable), DType::Primitive(PType::I32, Nullability::NonNullable)],
    "expects 3 argument dtypes, got 2",
)]
#[case::mismatched_then_else(
    CaseWhenOptions { num_when_then_pairs: 1, has_else: true },
    vec![
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::NonNullable),
    ],
    "THEN and ELSE dtypes must match (ignoring nullability)",
)]
#[case::mismatched_nary_then(
    CaseWhenOptions { num_when_then_pairs: 2, has_else: true },
    vec![
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Bool(Nullability::NonNullable),
        DType::Utf8(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ],
    "THEN dtypes must match",
)]
fn test_return_dtype_error_guards(
    #[case] options: CaseWhenOptions,
    #[case] arg_dtypes: Vec<DType>,
    #[case] expected_msg: &str,
) {
    let err = CaseWhen.return_dtype(&options, &arg_dtypes).unwrap_err();
    assert!(err.to_string().contains(expected_msg), "{err}");
}

#[rstest]
#[case::differing_operands(case_when(is_null(col("x")), lit(0i64), col("y")))]
#[case::non_constant_fill(case_when(is_null(col("x")), col("c"), col("x")))]
#[case::no_else(case_when_no_else(is_null(col("x")), lit(0i64)))]
#[case::multi_pair(nested_case_when(
    vec![
        (is_null(col("x")), lit(0i64)),
        (gt(col("x"), lit(5i64)), lit(1i64)),
    ],
    Some(col("x")),
))]
fn test_simplify_negative_guards_keep_case_when(#[case] expr: Expression) -> VortexResult<()> {
    let scope = DType::Struct(
        StructFields::new(
            ["x", "y", "c"].into(),
            vec![DType::Primitive(PType::I64, Nullability::Nullable); 3],
        ),
        Nullability::NonNullable,
    );
    let optimized = expr.bind(&scope)?.optimize_recursive()?;
    assert!(
        optimized.is::<CaseWhen>(),
        "expected CaseWhen to remain, got {optimized}"
    );
    Ok(())
}

#[rstest]
#[case::with_else(
    case_when(gt(col("value"), lit(0i32)), lit(100i32), lit(0i32)),
    "CASE WHEN ($.value > 0i32) THEN 100i32 ELSE 0i32 END"
)]
#[case::no_else(
    case_when_no_else(gt(col("value"), lit(0i32)), lit(100i32)),
    "CASE WHEN ($.value > 0i32) THEN 100i32 END"
)]
#[case::multi_pair(
    nested_case_when(
        vec![
            (gt(col("x"), lit(10i32)), lit("high")),
            (gt(col("x"), lit(5i32)), lit("medium")),
        ],
        Some(lit("low")),
    ),
    "CASE WHEN ($.x > 10i32) THEN \"high\" WHEN ($.x > 5i32) THEN \"medium\" ELSE \"low\" END"
)]
fn test_fmt_sql(#[case] expr: Expression, #[case] expected: &str) {
    assert_eq!(expr.to_string(), expected);
}
