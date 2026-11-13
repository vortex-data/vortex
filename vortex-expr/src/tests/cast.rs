// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::Expression;
use crate::cast;
use crate::get_item;
use crate::lit;
use crate::root;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();
    assert_eq!(
        cast(root(), DType::Bool(Nullability::NonNullable))
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::NonNullable)
    );
}

#[test]
fn replace_children() -> VortexResult<()> {
    let dtype = DType::Bool(Nullability::Nullable);
    let expr = cast(root(), dtype.clone());
    let replaced = expr.with_children(vec![get_item("a", root())])?;
    assert_eq!(replaced, cast(get_item("a", root()), dtype));
    Ok(())
}

#[test]
fn evaluate() {
    let test_array = StructArray::from_fields(&[
        ("a", buffer![0i32, 1, 2].into_array()),
        ("b", buffer![4i64, 5, 6].into_array()),
    ])
    .unwrap()
    .into_array();

    let expr: Expression = cast(
        get_item("a", root()),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    );
    let result = test_array.apply(&expr).unwrap();

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::I64, Nullability::NonNullable)
    );
    assert_arrays_eq!(
        result,
        buffer![0i64, 1, 2].into_array(),
        &mut SESSION.create_execution_ctx()
    );
}

#[test]
fn simplify_folds_cast_of_literal() -> VortexResult<()> {
    let expr = cast(
        lit(3i32),
        DType::Primitive(PType::F64, Nullability::NonNullable),
    );
    let optimized = expr.bind(&test_harness::struct_dtype())?.optimize()?;

    let scalar = optimized
        .as_opt::<Literal>()
        .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
    assert_eq!(scalar, &Scalar::primitive(3.0f64, Nullability::NonNullable));
    Ok(())
}

#[test]
fn simplify_folds_cast_of_decimal_literal() -> VortexResult<()> {
    let decimal = Scalar::decimal(
        DecimalValue::I128(319),
        DecimalDType::new(3, 2),
        Nullability::NonNullable,
    );
    let expr = cast(
        lit(decimal),
        DType::Primitive(PType::F64, Nullability::NonNullable),
    );
    let optimized = expr.bind(&test_harness::struct_dtype())?.optimize()?;

    let scalar = optimized
        .as_opt::<Literal>()
        .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
    assert_eq!(
        scalar,
        &Scalar::primitive(3.19f64, Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn simplify_leaves_failing_cast_unchanged() -> VortexResult<()> {
    let target = DType::Primitive(PType::F64, Nullability::NonNullable);
    let expr = cast(
        lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        ))),
        target.clone(),
    );
    let optimized = expr.bind(&test_harness::struct_dtype())?.optimize()?;

    assert!(optimized.as_opt::<Literal>().is_none());
    assert_eq!(optimized.as_opt::<Cast>(), Some(&target));
    Ok(())
}

#[test]
fn test_display() {
    let expr = cast(
        get_item("value", root()),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    );
    assert_eq!(expr.to_string(), "cast($.value as i64)");

    let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
    assert_eq!(expr2.to_string(), "cast($ as bool?)");
}
