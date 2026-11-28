// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::matcher::Matcher;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::scalar::NumericOperator;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::fill_null::*;
use vortex_array::scalar_fn::builtin::numeric::Numeric;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::fill_null;
use crate::get_item;
use crate::lit;
use crate::root;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_fill_null_rejects_other_base_type_for_non_nullable_input() -> VortexResult<()> {
    let input = buffer![1i32].into_array();
    let fill = Scalar::from(0i64);

    let expression_error = fill_null(root(), lit(fill.clone()))
        .bind(input.dtype())
        .expect_err("the input and fill have different base types");
    let array_error = input
        .fill_null(fill)
        .expect_err("the array shortcut must enforce the expression's dtype rule");

    // Compare only the message, because the error text includes a backtrace when
    // `RUST_BACKTRACE` is set, and the two paths fail at different call sites.
    let rule = "fill_null requires input and fill value to have the same base type";
    assert!(
        expression_error.to_string().contains(rule),
        "{expression_error}"
    );
    assert!(array_error.to_string().contains(rule), "{array_error}");
    Ok(())
}

/// A fill of `40 + 2` that is not a [`ConstantArray`] until it executes.
fn lazy_fill_42(len: usize) -> VortexResult<ArrayRef> {
    Ok(ScalarFnArray::try_new(
        Numeric.bind(NumericOperator::Add),
        vec![
            ConstantArray::new(40i32, len).into_array(),
            ConstantArray::new(2i32, len).into_array(),
        ],
    )?
    .into_array())
}

#[test]
fn dtype() {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    assert_eq!(
        fill_null(root(), lit(0i32)).return_dtype(&dtype).unwrap(),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    );
}

#[test]
fn replace_children() {
    let expr = fill_null(root(), lit(0i32));
    let replaced = expr
        .with_children(vec![get_item("a", root()), lit(42i32)])
        .vortex_expect("operation should succeed in test");
    assert_eq!(replaced, fill_null(get_item("a", root()), lit(42i32)));
}

#[test]
fn evaluate() {
    let mut ctx = SESSION.create_execution_ctx();
    let test_array =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array();

    let expr = fill_null(root(), lit(42i32));
    let result = test_array.apply(&expr).unwrap();

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([1i32, 42, 3, 42, 5]),
        &mut ctx
    );
}

#[test]
fn evaluate_lazy_constant_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let fill = lazy_fill_42(input.len())?;
    assert!(fill.as_constant().is_none());

    let result = FillNull::try_new(input, fill)?
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?;

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 42, 3]), &mut ctx);
    Ok(())
}

#[test]
fn evaluate_lazy_fill_preserves_chunked_input() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(3), None]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    )?
    .into_array();
    let fill = lazy_fill_42(input.len())?;

    let result = FillNull::try_new(input, fill)?
        .into_array()
        .execute::<ArrayRef>(&mut ctx)?;

    assert!(result.is::<Chunked>());
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([1i32, 42, 3, 42]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn non_constant_fill_declines_and_errors() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let varying = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

    let constant_input = ConstantArray::new(Scalar::null_native::<i32>(), 3).into_array();
    let constant_parent = FillNull::try_new(constant_input.clone(), varying.clone())?.into_array();
    let reduced = FillNullReduceAdaptor(Constant).reduce_parent(
        constant_input
            .as_opt::<Constant>()
            .vortex_expect("constant input"),
        ExactScalarFn::<FillNull>::try_match(&constant_parent).vortex_expect("fill_null parent"),
        0,
    )?;
    assert!(reduced.is_none());

    let primitive_input =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let primitive_parent = FillNull::try_new(primitive_input.clone(), varying)?.into_array();
    let executed = FillNullExecuteAdaptor(Primitive).execute_parent(
        primitive_input
            .as_opt::<Primitive>()
            .vortex_expect("primitive input"),
        ExactScalarFn::<FillNull>::try_match(&primitive_parent).vortex_expect("fill_null parent"),
        0,
        &mut ctx,
    )?;
    assert!(executed.is_none());

    let err = primitive_parent.execute::<ArrayRef>(&mut ctx).unwrap_err();
    assert!(
        err.to_string()
            .contains("fill_null fill_value must be a constant/scalar")
    );
    Ok(())
}

/// Filling a chunked array slices the constant fill per chunk, and an empty slice is a
/// canonical array rather than a constant. Stacked fills over an empty chunk must still run.
#[test]
fn repeated_fill_over_empty_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter(Vec::<Option<u16>>::new()).into_array(),
            PrimitiveArray::from_option_iter([Some(7u16), None]).into_array(),
        ],
        DType::Primitive(PType::U16, Nullability::Nullable),
    )?
    .into_array();

    let result = input
        .fill_null(Scalar::primitive(1u16, Nullability::Nullable))?
        .fill_null(Scalar::primitive(2u16, Nullability::NonNullable))?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([7u16, 1]), &mut ctx);
    Ok(())
}

#[test]
fn evaluate_struct_field() {
    let mut ctx = SESSION.create_execution_ctx();
    let test_array = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
    )])
    .unwrap()
    .into_array();

    let expr = fill_null(get_item("a", root()), lit(0i32));
    let result = test_array.apply(&expr).unwrap();

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 0, 3]), &mut ctx);
}

#[test]
fn evaluate_non_nullable_input() {
    let mut ctx = SESSION.create_execution_ctx();
    let test_array = buffer![1i32, 2, 3].into_array();
    let expr = fill_null(root(), lit(0i32));
    let result = test_array.apply(&expr).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);
}

/// Inputs with both valid and null rows execute for every non-zip dtype that `supports_dtype`
/// accepts, with a nullable or a non-nullable fill value.
#[rstest]
#[case::bool(DType::Bool(Nullability::Nullable))]
#[case::primitive(DType::Primitive(PType::I32, Nullability::Nullable))]
#[case::decimal(DType::Decimal(DecimalDType::new(19, 2), Nullability::Nullable))]
#[case::utf8(DType::Utf8(Nullability::Nullable))]
#[case::binary(DType::Binary(Nullability::Nullable))]
#[case::date(DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()))]
fn supported_dtypes_fill_mixed_validity(
    #[case] dtype: DType,
    #[values(Nullability::NonNullable, Nullability::Nullable)] fill_nullability: Nullability,
) -> VortexResult<()> {
    assert!(FillNull::supports_dtype(&dtype), "{dtype}");

    let mut ctx = SESSION.create_execution_ctx();
    let fill = Scalar::default_value(&dtype.as_nonnullable())
        .cast(&dtype.with_nullability(fill_nullability))?;
    let mut builder = builder_with_capacity_in(&dtype, 2, ctx.allocator());
    builder.append_scalar(&fill.cast(&dtype)?)?;
    builder.append_null();
    let filled = builder
        .finish()
        .fill_null(fill.clone())?
        .execute::<Canonical>(&mut ctx)?
        .into_array();

    assert_eq!(filled.dtype(), fill.dtype(), "{dtype}");
    assert_eq!(filled.execute_scalar(1, &mut ctx)?, fill, "{dtype}");
    assert_arrays_eq!(filled, ConstantArray::new(fill, 2), &mut ctx);
    Ok(())
}

#[rstest]
#[case::null(DType::Null)]
#[case::variant(DType::Variant(Nullability::Nullable))]
#[case::nested_variant(DType::struct_(
    [("value", DType::Variant(Nullability::Nullable))],
    Nullability::Nullable,
))]
fn unsupported_dtypes(#[case] dtype: DType) {
    assert!(!FillNull::supports_dtype(&dtype), "{dtype}");
}

/// A variant input with mixed validity fails with an error instead of a panic.
#[test]
fn variant_with_mixed_validity_errors() -> VortexResult<()> {
    let dtype = DType::Variant(Nullability::Nullable);
    let value = Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable));
    let core_storage = ChunkedArray::try_new(
        vec![
            ConstantArray::new(value.cast(&dtype)?, 1).into_array(),
            ConstantArray::new(Scalar::null(dtype.clone()), 1).into_array(),
        ],
        dtype,
    )?
    .into_array();
    let input = VariantArray::try_new(core_storage, None)?.into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let error = input
        .fill_null(value)?
        .execute::<Canonical>(&mut ctx)
        .unwrap_err();
    assert!(
        error.to_string().contains("fill_null is not supported"),
        "{error}"
    );
    Ok(())
}

#[test]
fn test_display() {
    let expr = fill_null(get_item("value", root()), lit(0i32));
    assert_eq!(expr.to_string(), "vortex.fill_null($.value, 0i32)");
}

/// A scalar function input is executed through the executor before its nulls are filled.
#[test]
fn executes_non_columnar_input_iteratively() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)])
        .into_array()
        .compare(
            ConstantArray::new(2i32, 3).into_array(),
            CompareOperator::Gt,
        )?;

    let result = input.fill_null(Scalar::from(false))?;

    assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]), &mut ctx);
    Ok(())
}
