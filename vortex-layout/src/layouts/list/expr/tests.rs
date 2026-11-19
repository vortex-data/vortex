// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::and;
use vortex_array::expr::checked_add;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::is_not_null;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::list_transform;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::var;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::ElementProjection;
use crate::test::SESSION;

/// `[{a, b}, {c, d}]`, `null` (hiding `[{e, i32::MAX}]`), `[]` and `[{null, 3}]`, as lists of
/// `{sku: utf8?, qty: i32}`.
fn items() -> VortexResult<ListArray> {
    let elements = StructArray::from_fields(&[
        (
            "sku",
            VarBinViewArray::from_iter_nullable_str([Some("a"), Some("c"), Some("e"), None])
                .into_array(),
        ),
        ("qty", buffer![1i32, 2, i32::MAX, 3].into_array()),
    ])?
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    ListArray::try_new(
        elements,
        buffer![0u32, 2, 3, 3, 4].into_array(),
        Validity::from_iter([true, false, true, true]),
        &mut ctx,
    )
}

fn sku(param: &str) -> VortexResult<Lambda> {
    Lambda::new([param], get_item("sku", var(param)))
}

fn qty(param: &str) -> VortexResult<Lambda> {
    Lambda::new([param], get_item("qty", var(param)))
}

/// `list_transform(root, x -> x.qty + 1)`, which overflows on the element of the null list.
fn qty_plus_one() -> VortexResult<Expression> {
    Ok(list_transform(
        root(),
        Lambda::new(["x"], checked_add(get_item("qty", var("x")), lit(1i32)))?,
    ))
}

/// Rewrites `expr` over the lists of [`items`].
fn push_down(expr: &Expression) -> VortexResult<Option<ElementProjection>> {
    ElementProjection::push_down(&expr.bind(items()?.dtype())?)
}

/// Evaluates a projection the way the reader does, over the whole of `list`.
fn evaluate(projection: &ElementProjection, list: &ListArray) -> VortexResult<ArrayRef> {
    let elements = list.elements().clone().apply_bound(projection.elements())?;
    // SAFETY: the offsets and validity are those of a valid list over the projected elements.
    let lists =
        unsafe { projection.rebuild(elements, list.offsets().clone(), list.list_validity()) }?;
    lists.apply_bound(projection.lists())
}

#[rstest]
#[case::one_field(list_transform(root(), sku("x")?), "$.sku", "$")]
#[case::param_name(list_transform(root(), sku("item")?), "$.sku", "$")]
#[case::predicate(
    and(
        is_not_null(root()),
        list_contains(list_transform(root(), sku("x")?), lit("c")),
    ),
    "$.sku",
    "(is_not_null($) and vortex.list.contains($, \"c\"))"
)]
#[case::length(
    pack(
        [("skus", list_transform(root(), sku("x")?)), ("len", list_length(root()))],
        Nullability::NonNullable,
    ),
    "$.sku",
    "pack(skus: $, len: vortex.list.length($))"
)]
#[case::repeated_lambda(
    pack(
        [("a", list_transform(root(), sku("x")?)), ("b", list_transform(root(), sku("x")?))],
        Nullability::NonNullable,
    ),
    "$.sku",
    "pack(a: $, b: $)"
)]
#[case::two_lambdas(
    pack(
        [
            ("skus", list_transform(root(), sku("x")?)),
            ("qtys", list_transform(root(), qty("x")?)),
            ("len", list_length(root())),
        ],
        Nullability::NonNullable,
    ),
    "pack(0: $.sku, 1: $.qty)",
    "pack(skus: $.0, qtys: $.1, len: vortex.list.length($.0))"
)]
#[case::outer_lambda_in_memory(
    list_transform(
        list_transform(root(), qty("x")?),
        Lambda::new(["y"], eq(var("y"), lit(1i32)))?,
    ),
    "$.qty",
    "vortex.list.transform($, (y) -> ($.y = 1i32))"
)]
fn pushes_lambdas_into_elements(
    #[case] expr: Expression,
    #[case] elements: &str,
    #[case] lists: &str,
) -> VortexResult<()> {
    let projection = push_down(&expr)?.expect("reads the elements only through lambdas");
    assert_eq!(projection.elements().to_string(), elements);
    assert_eq!(projection.lists().to_string(), lists);
    assert!(!projection.is_fallible());

    let list = items()?;
    let expected = list.clone().into_array().apply(&expr)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(evaluate(&projection, &list)?, expected, &mut ctx);
    Ok(())
}

/// A lambda that can fail makes the projection fallible, so the reader runs it only on the
/// elements of valid lists.
#[test]
fn fallible_lambda_is_fallible() -> VortexResult<()> {
    let projection = push_down(&qty_plus_one()?)?.expect("reads the elements through a lambda");
    assert_eq!(projection.elements().to_string(), "($.qty + 1i32)");
    assert!(projection.is_fallible());
    Ok(())
}

#[rstest]
#[case::root(root())]
#[case::no_lambda(list_length(root()))]
#[case::mixed(pack(
    [("items", root()), ("skus", list_transform(root(), sku("x")?))],
    Nullability::NonNullable,
))]
fn keeps_element_reads(#[case] expr: Expression) -> VortexResult<()> {
    assert!(push_down(&expr)?.is_none());
    Ok(())
}

/// The identity projection reads the elements as stored and evaluates the expression over the
/// rebuilt list.
#[test]
fn identity_reads_stored_elements() -> VortexResult<()> {
    let list = items()?;
    let expr = list_contains(list_transform(root(), qty("x")?), lit(3i32));
    let projection =
        ElementProjection::identity(&expr.bind(list.dtype())?, list.elements().dtype());
    assert!(projection.elements().is_root());
    assert!(!projection.is_fallible());

    let expected = list.clone().into_array().apply(&expr)?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(evaluate(&projection, &list)?, expected, &mut ctx);
    Ok(())
}

/// Lists of `i32` pushed into a flat elements child keep their shape.
#[test]
fn primitive_elements() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let list = ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        buffer![0u32, 0, 2, 3].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    let expr = list_transform(root(), Lambda::new(["x"], eq(var("x"), lit(3i32)))?);
    let projection = ElementProjection::push_down(&expr.bind(list.dtype())?)?
        .expect("reads the elements through a lambda");
    assert_eq!(projection.elements().to_string(), "($ = 3i32)");
    assert!(matches!(projection.lists().dtype(), DType::List(..)));

    let expected = list.clone().into_array().apply(&expr)?;
    assert_arrays_eq!(evaluate(&projection, &list)?, expected, &mut ctx);
    Ok(())
}
