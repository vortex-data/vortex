// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::matcher::Matcher;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::fill_null::*;
use vortex_array::scalar_fn::builtin::operators::Operator;
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

/// A fill of `40 + 2` that is not a [`ConstantArray`] until it executes.
fn lazy_fill_42(len: usize) -> VortexResult<ArrayRef> {
    Ok(ScalarFnArray::try_new(
        Binary.bind(Operator::Add),
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

/// Inputs with both valid and null rows execute exactly for the dtypes that `supports_dtype`
/// accepts, so adding a kernel without updating `supports_dtype` fails here.
#[test]
fn supports_dtype_matches_kernels() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let dtypes = [
        DType::Bool(Nullability::Nullable),
        DType::Primitive(PType::I32, Nullability::Nullable),
        DType::Decimal(DecimalDType::new(19, 2), Nullability::Nullable),
        DType::Utf8(Nullability::Nullable),
        DType::Binary(Nullability::Nullable),
        DType::List(Arc::new(i32_dtype.clone()), Nullability::Nullable),
        DType::FixedSizeList(Arc::new(i32_dtype.clone()), 2, Nullability::Nullable),
        DType::Struct(
            StructFields::from_iter([("value", i32_dtype.clone())]),
            Nullability::Nullable,
        ),
        DType::Map(
            MapDType::try_new(i32_dtype.clone(), i32_dtype, false)?,
            Nullability::Nullable,
        ),
        DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()),
    ];

    let supported = [
        true, true, true, true, true, false, false, false, false, false,
    ];
    for (dtype, supported) in dtypes.into_iter().zip(supported) {
        assert_eq!(FillNull::supports_dtype(&dtype), supported, "{dtype}");
        let fill = Scalar::default_value(&dtype.as_nonnullable());
        let mut builder = builder_with_capacity_in(&dtype, 2, ctx.allocator());
        builder.append_scalar(&fill.cast(&dtype)?)?;
        builder.append_null();
        let filled = builder
            .finish()
            .fill_null(fill.clone())?
            .execute::<Canonical>(&mut ctx);

        if supported {
            let filled = filled?.into_array();
            assert_eq!(filled.dtype(), fill.dtype(), "{dtype}");
            assert_eq!(filled.execute_scalar(1, &mut ctx)?, fill, "{dtype}");
            assert_arrays_eq!(filled, ConstantArray::new(fill, 2), &mut ctx);
        } else {
            assert!(filled.is_err(), "{dtype} executes but is not supported");
        }
    }
    Ok(())
}

#[test]
fn test_display() {
    let expr = fill_null(get_item("value", root()), lit(0i32));
    assert_eq!(expr.to_string(), "vortex.fill_null($.value, 0i32)");
}
