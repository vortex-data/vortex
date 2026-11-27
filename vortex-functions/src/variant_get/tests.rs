// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar_is_null;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::proto::ExprSerializeProtoExt;
use vortex_expr::root;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::expr::variant_get;
use crate::variant_get::VariantGet;
use crate::variant_get::VariantGetOptions;
use crate::variant_get::VariantPath;
use crate::variant_get::VariantPathElement;
use crate::variant_get::convert_to_decimal;
use crate::variant_pack::VariantPack;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();

    vortex_array::initialize(&session);

    crate::initialize(&session);

    session
});

pub(super) fn variant_object(fields: impl IntoIterator<Item = (&'static str, Scalar)>) -> Scalar {
    let fields = fields.into_iter().collect::<Vec<_>>();
    let names = FieldNames::from_iter(fields.iter().map(|(name, _)| FieldName::from(*name)));
    let dtypes = vec![DType::Variant(Nullability::NonNullable); fields.len()];
    let values = fields
        .into_iter()
        .map(|(_, value)| Scalar::variant(value).into_value())
        .collect();
    Scalar::try_new(
        DType::Struct(StructFields::new(names, dtypes), Nullability::NonNullable),
        Some(ScalarValue::Tuple(values)),
    )
    .unwrap()
}

pub(super) fn variant_rows(rows: impl IntoIterator<Item = Scalar>) -> VortexResult<ArrayRef> {
    let dtype = DType::Variant(Nullability::Nullable);
    let chunks = rows
        .into_iter()
        .map(|row| ConstantArray::new(row.cast(&dtype).unwrap(), 1).into_array());
    ChunkedArray::try_new(chunks, dtype.clone()).map(|array| array.into_array())
}

/// Test-only syntax for keeping `variant_get` cases compact without committing
/// to a public string grammar yet.
fn parse_path(path: &str) -> VortexResult<VariantPath> {
    if path.is_empty() || path == "$" {
        return Ok(VariantPath::root());
    }

    let mut elements = Vec::new();
    let mut pos = usize::from(path.as_bytes().first() == Some(&b'$'));
    if pos == 1
        && path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| !matches!(byte, b'.' | b'['))
    {
        vortex_bail!("Invalid Variant path {path:?}: expected '.' or '[' after '$'");
    }

    while pos < path.len() {
        match path.as_bytes()[pos] {
            b'.' => {
                pos += 1;
                let (field, next_pos) = parse_field(path, pos)?;
                elements.push(VariantPathElement::field(field));
                pos = next_pos;
            }
            b'[' => {
                let (index, next_pos) = parse_index(path, pos + 1)?;
                elements.push(VariantPathElement::index(index));
                pos = next_pos;
            }
            _ if pos == 0 => {
                let (field, next_pos) = parse_field(path, pos)?;
                elements.push(VariantPathElement::field(field));
                pos = next_pos;
            }
            _ => {
                vortex_bail!("Invalid Variant path {path:?}: expected '.', '[', or end of path")
            }
        }
    }

    Ok(VariantPath::new(elements))
}

fn parse_field(path: &str, start: usize) -> VortexResult<(&str, usize)> {
    let mut pos = start;
    while path
        .as_bytes()
        .get(pos)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        pos += 1;
    }
    vortex_ensure!(
        pos > start,
        "Invalid Variant path {path:?}: expected field name"
    );
    Ok((&path[start..pos], pos))
}

fn parse_index(path: &str, start: usize) -> VortexResult<(u64, usize)> {
    let mut pos = start;
    while path
        .as_bytes()
        .get(pos)
        .is_some_and(|byte| byte.is_ascii_digit())
    {
        pos += 1;
    }
    vortex_ensure!(
        pos > start,
        "Invalid Variant path {path:?}: expected list index"
    );
    vortex_ensure!(
        path.as_bytes().get(pos) == Some(&b']'),
        "Invalid Variant path {path:?}: expected closing ']'"
    );
    let index = path[start..pos]
        .parse()
        .map_err(|_| vortex_err!("Invalid Variant path {path:?}: list index is too large"))?;
    Ok((index, pos + 1))
}

fn execute_variant_get(
    array: ArrayRef,
    path: &str,
    dtype: Option<DType>,
) -> VortexResult<ArrayRef> {
    let expr = variant_get(root(), parse_path(path)?, dtype);
    array
        .apply(&expr)?
        .optimize_ctx(&SESSION)?
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
}

#[test]
fn variant_get_path_parse_and_display() {
    let path = parse_path("$.data[1].a").unwrap();
    assert_eq!(
        path.elements(),
        &[
            VariantPathElement::field("data"),
            VariantPathElement::index(1),
            VariantPathElement::field("a")
        ]
    );
    assert_eq!(path.to_string(), "$.data[1].a");

    let bare_path = parse_path("data[2]").unwrap();
    assert_eq!(bare_path.to_string(), "$.data[2]");
}

#[test]
fn variant_get_return_dtype_is_nullable_variant_without_requested_dtype() {
    let expr = variant_get(root(), VariantPath::field("data"), None);
    let dtype = expr
        .return_dtype(&DType::Variant(Nullability::NonNullable))
        .unwrap();

    assert_eq!(dtype, DType::Variant(Nullability::Nullable));
}

#[test]
fn variant_get_return_dtype_makes_requested_dtype_nullable() {
    let requested = DType::Primitive(PType::I64, Nullability::NonNullable);
    let expr = variant_get(root(), VariantPath::field("data"), Some(requested));
    let dtype = expr
        .return_dtype(&DType::Variant(Nullability::NonNullable))
        .unwrap();

    assert_eq!(dtype, DType::Primitive(PType::I64, Nullability::Nullable));
}

#[test]
fn variant_get_rejects_non_variant_input() {
    let expr = variant_get(root(), VariantPath::field("data"), None);
    let err = expr
        .return_dtype(&DType::Utf8(Nullability::NonNullable))
        .unwrap_err();

    assert!(err.to_string().contains("VariantGet input must be Variant"));
}

#[test]
fn variant_get_formats_sql() {
    let expr = variant_get(
        root(),
        parse_path("$.data[1].a").unwrap(),
        Some(DType::Utf8(Nullability::NonNullable)),
    );

    assert_eq!(expr.to_string(), "variant_get($, \"$.data[1].a\", utf8)");
}

#[test]
fn variant_get_options_roundtrip_serialization() {
    let options = VariantGetOptions::new(
        VariantPath::new([
            VariantPathElement::field("data"),
            VariantPathElement::index(1),
            VariantPathElement::field("a"),
        ]),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let metadata = VariantGet.serialize(&options).unwrap().unwrap();
    let actual = VariantGet.deserialize(&metadata, &SESSION).unwrap();

    assert_eq!(actual, options);
}

#[test]
fn variant_get_expression_roundtrip_serialization() {
    let expr: Expression = variant_get(
        root(),
        parse_path("$.data[1].a").unwrap(),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let proto = expr.serialize_proto().unwrap();
    let actual = Expression::from_proto(&proto, &SESSION).unwrap();

    assert_eq!(actual, expr);
}

#[test]
fn variant_get_generic_fallback_extracts_field_and_list_index() -> VortexResult<()> {
    let items = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    );
    let array = variant_rows([
        Scalar::variant(variant_object([("items", items)])),
        Scalar::variant(variant_object([(
            "items",
            Scalar::list_empty(
                DType::Variant(Nullability::NonNullable).into(),
                Nullability::NonNullable,
            ),
        )])),
        Scalar::variant(variant_object([(
            "items",
            Scalar::list(
                DType::Variant(Nullability::NonNullable),
                vec![
                    Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
                    Scalar::variant(Scalar::utf8("wrong", Nullability::NonNullable)),
                ],
                Nullability::NonNullable,
            ),
        )])),
    ])?;

    let result = execute_variant_get(
        array,
        "$.items[1]",
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    )?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(20i32), None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_reads_chunked_variant_input() -> VortexResult<()> {
    let array = variant_rows([
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(10i32, Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "b",
            Scalar::primitive(20i32, Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(30i32, Nullability::NonNullable),
        )])),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    assert!(array.is::<Chunked>());

    let result = execute_variant_get(
        array,
        "$.a",
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    )?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_fallback_typed_output_is_contiguous() -> VortexResult<()> {
    let array = variant_rows([
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(10i32, Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(20i32, Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "b",
            Scalar::primitive(30i32, Nullability::NonNullable),
        )])),
    ])?;

    let result = execute_variant_get(
        array,
        "$.a",
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    )?;

    assert!(!result.is::<Chunked>());
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_generic_fallback_preserves_variant_null() -> VortexResult<()> {
    let array = variant_rows([
        Scalar::variant(variant_object([(
            "a",
            Scalar::utf8("ok", Nullability::NonNullable),
        )])),
        Scalar::null(DType::Variant(Nullability::Nullable)),
        Scalar::variant(variant_object([("a", Scalar::null(DType::Null))])),
        Scalar::variant(variant_object([(
            "b",
            Scalar::primitive(2i32, Nullability::NonNullable),
        )])),
    ])?;

    let result = execute_variant_get(array, "$.a", None)?;

    let mut ctx = SESSION.create_execution_ctx();
    let row0 = result.execute_scalar(0, &mut ctx)?;
    assert_eq!(
        row0.as_variant()
            .value()
            .and_then(|value| value.as_utf8().value())
            .map(|value| value.as_str()),
        Some("ok")
    );
    assert_nth_scalar_is_null!(result, 1, &mut ctx);
    assert_eq!(
        result
            .execute_scalar(2, &mut ctx)?
            .as_variant()
            .is_variant_null(),
        Some(true)
    );
    assert_nth_scalar_is_null!(result, 3, &mut ctx);
    Ok(())
}

#[test]
fn variant_get_fallback_variant_output_canonicalizes() -> VortexResult<()> {
    let array = variant_rows([
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(10i32, Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "a",
            Scalar::primitive(20i32, Nullability::NonNullable),
        )])),
    ])?;

    let result = execute_variant_get(array, "$.a", None)?;
    let variant = result
        .clone()
        .execute::<VariantArray>(&mut SESSION.create_execution_ctx())?;
    let canonical = result.execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
    let Canonical::Variant(canonical_variant) = canonical else {
        vortex_bail!("expected Variant canonical array");
    };

    assert_eq!(variant.len(), 2);
    assert_eq!(canonical_variant.len(), 2);
    assert_eq!(variant.core_storage().dtype(), variant.dtype());
    assert_eq!(variant.core_storage().len(), variant.len());

    let mut ctx = SESSION.create_execution_ctx();
    for array in [variant, canonical_variant] {
        assert_eq!(array.dtype(), &DType::Variant(Nullability::Nullable));
        for (idx, expected) in [10i32, 20].into_iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let actual = scalar
                .as_variant()
                .value()
                .and_then(|value| value.as_primitive().as_::<i32>());
            assert_eq!(actual, Some(expected), "row {idx}");
        }
    }
    Ok(())
}

fn row_storage(values: impl IntoIterator<Item = i32>) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(
        values.into_iter().map(|value| {
            ConstantArray::new(
                Scalar::variant(Scalar::primitive(value, Nullability::NonNullable)),
                1,
            )
            .into_array()
        }),
        DType::Variant(Nullability::NonNullable),
    )?
    .into_array())
}

#[test]
fn variant_get_keeps_valid_shredded_rows_for_matching_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = row_storage([1, 2, 3])?;
    let shredded = StructArray::try_from_iter([(
        "a",
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
    )])?;
    let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;
    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );

    let result = variant.into_array().apply(&expr)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_treats_value_and_typed_value_as_logical_field_names() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = row_storage([1, 2, 3])?;
    let shredded = StructArray::try_from_iter([
        (
            "value",
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        ),
        (
            "typed_value",
            PrimitiveArray::from_iter([40i32, 50, 60]).into_array(),
        ),
    ])?;
    let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;

    let value_expr = variant_get(
        root(),
        VariantPath::field("value"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let value_result = variant.clone().into_array().apply(&value_expr)?;
    assert_arrays_eq!(
        value_result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
        &mut ctx
    );

    let typed_value_expr = variant_get(
        root(),
        VariantPath::field("typed_value"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let typed_value_result = variant.into_array().apply(&typed_value_expr)?;
    assert_arrays_eq!(
        typed_value_result,
        PrimitiveArray::from_option_iter([Some(40i32), Some(50), Some(60)]),
        &mut ctx
    );
    Ok(())
}

fn obj(name: &'static str, value: Scalar) -> Scalar {
    Scalar::variant(variant_object([(name, value)]))
}

fn obj_a(value: i32) -> Scalar {
    obj("a", Scalar::primitive(value, Nullability::NonNullable))
}

#[test]
fn variant_get_propagates_null_nested_shredded_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows([
        obj(
            "nested",
            variant_object([("a", Scalar::primitive(100i32, Nullability::NonNullable))]),
        ),
        obj(
            "nested",
            variant_object([("a", Scalar::primitive(200i32, Nullability::NonNullable))]),
        ),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    let nested = StructArray::try_new(
        FieldNames::from(["a"]),
        vec![PrimitiveArray::from_iter([10i32, 20, 30]).into_array()],
        3,
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let shredded = StructArray::try_from_iter([("nested", nested)])?.into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let expr = variant_get(
        root(),
        parse_path("$.nested.a")?,
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(200), None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::missing_struct_field("$.missing", true, [Some(100i32), Some(200), None])]
#[case::list_index_path("$.items[1]", true, [Some(20i32), None, None])]
#[case::field_through_non_struct("$.a.b", true, [None, None, None])]
#[case::no_shredded_child("$.missing", false, [Some(100i32), Some(200), None])]
fn variant_get_shredded_none_path_falls_back_to_core(
    #[case] path: &str,
    #[case] with_shredded: bool,
    #[case] expected: [Option<i32>; 3],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let list_two = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    );
    let list_one = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![Scalar::variant(Scalar::primitive(
            30i32,
            Nullability::NonNullable,
        ))],
        Nullability::NonNullable,
    );
    let core_storage = variant_rows([
        Scalar::variant(variant_object([
            ("a", Scalar::primitive(1i32, Nullability::NonNullable)),
            (
                "missing",
                Scalar::primitive(100i32, Nullability::NonNullable),
            ),
            ("items", list_two),
        ])),
        Scalar::variant(variant_object([
            ("a", Scalar::primitive(2i32, Nullability::NonNullable)),
            (
                "missing",
                Scalar::primitive(200i32, Nullability::NonNullable),
            ),
            ("items", list_one),
        ])),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    let shredded = with_shredded
        .then(|| {
            let items = ListArray::try_new(
                buffer![10i32, 20, 30].into_array(),
                buffer![0u32, 2, 3, 3].into_array(),
                Validity::NonNullable,
                &mut ctx,
            )?
            .into_array();
            Ok::<_, vortex_error::VortexError>(
                StructArray::try_from_iter([
                    ("a", buffer![1i32, 2, 3].into_array()),
                    ("items", items),
                ])?
                .into_array(),
            )
        })
        .transpose()?;
    let variant = VariantArray::try_new(core_storage, shredded)?.into_array();

    let expr = variant_get(
        root(),
        parse_path(path)?,
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid_variant_child(
    vec![
        Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
        Scalar::variant(Scalar::utf8("not_an_int", Nullability::NonNullable)),
        Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
    ],
    [Some(10i32), None, Some(30)],
)]
#[case::null_variant_child_falls_back_to_core(
    vec![
        Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
        Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
    ],
    [Some(10i32), Some(200), Some(30)],
)]
#[case::all_null_variant_child_falls_back_to_core(
    vec![
        Scalar::null(DType::Variant(Nullability::Nullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ],
    [Some(100i32), Some(200), Some(300)],
)]
fn variant_get_shredded_variant_child_casts_to_concrete_dtype(
    #[case] shredded_rows: Vec<Scalar>,
    #[case] expected: [Option<i32>; 3],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows([obj_a(100), obj_a(200), obj_a(300)])?;
    let shredded_variant = VariantArray::try_new(variant_rows(shredded_rows)?, None)?;
    let shredded = StructArray::try_from_iter([("a", shredded_variant.into_array())])?.into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
    Ok(())
}

#[rstest]
#[case::none_dtype(None)]
#[case::explicit_variant_dtype(Some(DType::Variant(Nullability::NonNullable)))]
fn variant_get_untyped_merges_shredded_struct_with_fallback(
    #[case] requested_dtype: Option<DType>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows([
        obj(
            "obj",
            variant_object([
                ("a", Scalar::primitive(1i32, Nullability::NonNullable)),
                ("b", Scalar::utf8("raw0", Nullability::NonNullable)),
            ]),
        ),
        obj(
            "obj",
            variant_object([("b", Scalar::utf8("raw1", Nullability::NonNullable))]),
        ),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    let shredded_obj = StructArray::try_from_iter([(
        "a",
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
    )])?
    .into_array();
    let shredded = StructArray::try_from_iter([("obj", shredded_obj)])?.into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let expr = variant_get(root(), VariantPath::field("obj"), requested_dtype);
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        variant_rows([
            Scalar::variant(variant_object([
                ("a", Scalar::primitive(10i32, Nullability::NonNullable)),
                ("b", Scalar::utf8("raw0", Nullability::NonNullable)),
            ])),
            obj("b", Scalar::utf8("raw1", Nullability::NonNullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?,
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::all_valid_primitive(
    PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
    vec![obj_a(1), obj_a(2), obj_a(3)],
    vec![
        Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
        Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
        Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
    ],
)]
#[case::partial_null_primitive(
    PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
    vec![
        obj_a(1),
        obj("a", Scalar::utf8("fallback", Nullability::NonNullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ],
    vec![
        Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
        Scalar::variant(Scalar::utf8("fallback", Nullability::NonNullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ],
)]
fn variant_get_untyped_merges_shredded_leaf_values(
    #[case] shredded_a: ArrayRef,
    #[case] core_rows: Vec<Scalar>,
    #[case] expected: Vec<Scalar>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows(core_rows)?;
    let shredded = StructArray::try_from_iter([("a", shredded_a)])?.into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let expr = variant_get(root(), VariantPath::field("a"), None);
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(result, variant_rows(expected)?, &mut ctx);
    Ok(())
}

/// A typed row reads from the shredded child, and a null typed row from core storage.
#[rstest]
#[case::list(false)]
#[case::fixed_size_list(true)]
fn variant_get_untyped_list_reads_typed_and_core_rows(
    #[case] fixed_size: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let no_value = Scalar::variant(Scalar::null(DType::Null));
    let core_storage = variant_rows([
        no_value,
        Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
    ])?;
    let validity = Validity::from_iter([true, false]);
    let shredded_items = if fixed_size {
        FixedSizeListArray::try_new(buffer![1i32, 2, 0, 0].into_array(), 2, validity, 2)?
            .into_array()
    } else {
        ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 2, 2].into_array(),
            validity,
            &mut ctx,
        )?
        .into_array()
    };
    let variant = VariantArray::try_new(core_storage, Some(shredded_items))?.into_array();

    let typed_list = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(2i32, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    );

    let expr = variant_get(root(), VariantPath::root(), None);
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        variant_rows([
            Scalar::variant(typed_list),
            Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
        ])?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_converts_typed_values_of_another_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows([
        obj_a(10),
        obj_a(20),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    let shredded = StructArray::try_from_iter([(
        "a",
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
    )])?
    .into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let widened = variant.clone().apply(&variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I64, Nullability::NonNullable)),
    ))?;
    assert_arrays_eq!(
        widened,
        PrimitiveArray::from_option_iter([Some(10i64), Some(20), None]),
        &mut ctx
    );

    let incompatible = variant.apply(&variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Utf8(Nullability::NonNullable)),
    ))?;
    assert_arrays_eq!(
        incompatible,
        VarBinViewArray::from_iter_nullable_str([None::<&str>, None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_partial_null_shredded_rows_zip_with_fallback() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core_storage = variant_rows([
        obj_a(100),
        obj_a(200),
        obj("a", Scalar::utf8("not_int", Nullability::NonNullable)),
        Scalar::null(DType::Variant(Nullability::Nullable)),
    ])?;
    let shredded = StructArray::try_from_iter([(
        "a",
        PrimitiveArray::from_option_iter([Some(10i32), None, None, Some(40)]).into_array(),
    )])?
    .into_array();
    let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(200), None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_unwraps_nested_variant_core_storage() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner_core = variant_rows([obj_a(10), obj_a(20)])?;
    let inner_variant = VariantArray::try_new(inner_core, None)?.into_array();
    let outer_variant = VariantArray::try_new(inner_variant, None)?.into_array();

    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
    );
    let result = outer_variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(10i32), Some(20)]),
        &mut ctx
    );
    Ok(())
}

fn variant_list(elements: impl IntoIterator<Item = Scalar>) -> Scalar {
    Scalar::list(
        DType::Variant(Nullability::NonNullable),
        elements.into_iter().map(Scalar::variant).collect(),
        Nullability::NonNullable,
    )
}

#[test]
fn variant_get_non_nullable_nested_target_returns_null_without_panicking() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = variant_rows([
        Scalar::variant(variant_object([(
            "x",
            Scalar::utf8("not_an_int", Nullability::NonNullable),
        )])),
        Scalar::variant(variant_object([(
            "x",
            Scalar::primitive(7i32, Nullability::NonNullable),
        )])),
    ])?;

    let struct_target = DType::Struct(
        StructFields::new(
            FieldNames::from(["x"]),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::Nullable,
    );
    let expr = variant_get(root(), VariantPath::root(), Some(struct_target));
    let result = input.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
    assert!(result.is_invalid(0, &mut ctx)?);
    assert_eq!(
        result
            .execute_scalar(1, &mut ctx)?
            .as_struct()
            .field("x")
            .and_then(|s| s.as_primitive().typed_value::<i32>()),
        Some(7)
    );

    let list_input = variant_rows([
        Scalar::variant(variant_list([Scalar::utf8(
            "bad",
            Nullability::NonNullable,
        )])),
        Scalar::variant(variant_list([Scalar::primitive(
            9i32,
            Nullability::NonNullable,
        )])),
    ])?;

    let list_target = DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::Nullable,
    );
    let list_expr = variant_get(root(), VariantPath::root(), Some(list_target));
    let list_result = list_input
        .apply(&list_expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    assert!(list_result.is_invalid(0, &mut ctx)?);
    assert_eq!(
        list_result
            .execute_scalar(1, &mut ctx)?
            .as_list()
            .element(0)
            .and_then(|s| s.as_primitive().typed_value::<i32>()),
        Some(9)
    );
    Ok(())
}

#[test]
fn variant_get_fixed_size_list_target() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = variant_rows([
        Scalar::variant(variant_list([
            Scalar::primitive(1i64, Nullability::NonNullable),
            Scalar::primitive(2i64, Nullability::NonNullable),
        ])),
        Scalar::variant(variant_list([Scalar::primitive(
            3i64,
            Nullability::NonNullable,
        )])),
    ])?;

    let target = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        2,
        Nullability::Nullable,
    );
    let expr = variant_get(root(), VariantPath::root(), Some(target));
    let result = input.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
    let scalar0 = result.execute_scalar(0, &mut ctx)?;
    assert_eq!(
        scalar0.as_list().elements().map(|elems| elems
            .into_iter()
            .map(|e| e.as_primitive().typed_value::<i32>())
            .collect::<Vec<_>>()),
        Some(vec![Some(1), Some(2)])
    );
    assert!(result.is_invalid(1, &mut ctx)?);
    Ok(())
}

#[test]
fn variant_get_fixed_size_list_preserves_variant_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let row = Scalar::variant(variant_list([
        Scalar::primitive(7i32, Nullability::NonNullable),
        Scalar::null(DType::Null),
    ]));
    let input = ConstantArray::new(row, 1).into_array();

    for elem_nullability in [Nullability::NonNullable, Nullability::Nullable] {
        let target = DType::FixedSizeList(
            Arc::new(DType::Variant(elem_nullability)),
            2,
            Nullability::Nullable,
        );
        let expr = variant_get(root(), VariantPath::root(), Some(target));
        let result = input.clone().apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is_valid(0, &mut ctx)?);
        let list = result.execute_scalar(0, &mut ctx)?;
        let elems = list
            .as_list()
            .elements()
            .ok_or_else(|| vortex_error::vortex_err!("expected non-null fixed-size list"))?;
        assert_eq!(elems.len(), 2);
        assert_eq!(
            elems[0]
                .as_variant()
                .value()
                .and_then(|v| v.as_primitive().typed_value::<i32>()),
            Some(7)
        );
        assert!(!elems[1].is_null());
        assert_eq!(elems[1].as_variant().is_variant_null(), Some(true));
    }
    Ok(())
}

/// The lazy shapes over a constant in [`variant_get_extracts_from_lazy_constants`].
#[derive(Clone, Copy, Debug)]
enum Lazy {
    Constant,
    Sliced,
    Filtered,
    Masked,
}

/// A constant canonicalizes to a Variant array over itself. Extraction from it and from the
/// lazy arrays over it must end, with the value of the constant in each row.
#[rstest]
#[case::constant(Lazy::Constant, [Some(5i32), Some(5), Some(5)])]
#[case::sliced(Lazy::Sliced, [Some(5i32), Some(5), Some(5)])]
#[case::filtered(Lazy::Filtered, [Some(5i32), Some(5), Some(5)])]
#[case::masked(Lazy::Masked, [Some(5i32), None, Some(5)])]
fn variant_get_extracts_from_lazy_constants(
    #[case] lazy: Lazy,
    #[case] expected: [Option<i32>; 3],
) -> VortexResult<()> {
    let row = Scalar::variant(variant_object([(
        "a",
        Scalar::primitive(5i32, Nullability::NonNullable),
    )]))
    .cast(&DType::Variant(Nullability::Nullable))?;
    let constant = ConstantArray::new(row.clone(), 4).into_array();
    let input = match lazy {
        Lazy::Constant => ConstantArray::new(row, 3).into_array(),
        Lazy::Sliced => constant.slice(1..4)?,
        Lazy::Filtered => constant.filter(Mask::from_iter([true, false, true, true]))?,
        Lazy::Masked => constant
            .slice(0..3)?
            .mask(BoolArray::from_iter([true, false, true]).into_array())?,
    };

    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::Nullable)),
    );
    let result = input
        .apply(&expr)?
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;

    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn converts_decimals_to_zero_when_every_digit_is_dropped() {
    let value = Scalar::decimal(
        DecimalValue::I64(99_999_999),
        DecimalDType::new(8, 3),
        Nullability::NonNullable,
    );
    let target = DecimalDType::new(6, -36);

    assert_eq!(
        convert_to_decimal(&value, &DType::Decimal(target, Nullability::Nullable)),
        Some(Scalar::decimal(
            DecimalValue::I128(0),
            target,
            Nullability::Nullable
        ))
    );
}

#[rstest]
#[case::rounds_up(12_345, Some(1_235))]
#[case::rounds_down(12_344, Some(1_234))]
#[case::rounds_negative_away_from_zero(-12_345, Some(-1_235))]
#[case::too_wide_for_precision(99_999_995, None)]
fn converts_decimals_to_a_smaller_scale(#[case] unscaled: i64, #[case] expected: Option<i64>) {
    let value = Scalar::decimal(
        DecimalValue::I64(unscaled),
        DecimalDType::new(8, 3),
        Nullability::NonNullable,
    );
    let target = DecimalDType::new(6, 2);

    assert_eq!(
        convert_to_decimal(&value, &DType::Decimal(target, Nullability::Nullable)),
        expected.map(|expected| Scalar::decimal(
            DecimalValue::I128(i128::from(expected)),
            target,
            Nullability::Nullable,
        ))
    );
}

#[rstest]
#[case::masked_input(false)]
#[case::masked_core_storage(true)]
fn variant_get_extracts_from_masked_variants(#[case] as_core_storage: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let row = obj_a(5).cast(&DType::Variant(Nullability::Nullable))?;
    let masked = MaskedArray::try_new(
        ConstantArray::new(row, 3).into_array(),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    )?
    .into_array();
    let input = if as_core_storage {
        VariantArray::try_new(masked, None)?.into_array()
    } else {
        masked
    };

    let expr = variant_get(
        root(),
        VariantPath::field("a"),
        Some(DType::Primitive(PType::I32, Nullability::Nullable)),
    );
    let result = input.apply(&expr)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5i32), None, Some(5)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_root_struct_respects_shredded_struct_validity() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let core = variant_rows([
        Scalar::variant(variant_object([])),
        Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
    ])?;
    let shredded = StructArray::try_new(
        FieldNames::from_iter(["a"]),
        vec![PrimitiveArray::from_option_iter([Some(10i32), Some(11)]).into_array()],
        2,
        Validity::from_iter([true, false]),
    )?
    .into_array();
    let packed = VariantPack::try_new(core, shredded, true)?.into_array();

    let target = DType::struct_(
        [("a", DType::Primitive(PType::I32, Nullability::Nullable))],
        Nullability::Nullable,
    );
    let expr = variant_get(root(), VariantPath::root(), Some(target));
    let result = packed
        .apply(&expr)?
        .optimize_ctx(&ctx)?
        .execute::<ArrayRef>(&mut ctx)?;
    assert!(result.is_valid(0, &mut ctx)?);
    assert_eq!(
        result
            .execute_scalar(0, &mut ctx)?
            .as_struct()
            .field("a")
            .and_then(|s| s.as_primitive().typed_value::<i32>()),
        Some(10),
    );
    assert!(result.is_invalid(1, &mut ctx)?);
    Ok(())
}

/// How [`VariantGet`] over a [`VariantPack`] reduces.
#[derive(Debug, Clone, Copy)]
enum Reduced {
    /// To a read of the shredded child alone.
    ToShredded,
    /// To an extraction from the core storage alone.
    ToCore,
    /// Not at all: the pack stays and executes to a whole Variant.
    No,
}

/// Variant objects whose field `a` is shredded as `i32` and field `n` as a shredded Variant,
/// over core storage that holds field `b`, and a last row that is a string, not an object. When
/// the pack is not complete, the core storage also holds `a` for the third row, where the
/// shredded `a` is null.
///
/// Returns the core storage and the shredded struct, which is null for the string row.
fn pack_over_core(complete: bool) -> VortexResult<(ArrayRef, ArrayRef)> {
    let b = |value: i32| Scalar::primitive(value, Nullability::NonNullable);
    let third_row = if complete {
        variant_object([("b", b(3))])
    } else {
        variant_object([
            ("a", Scalar::primitive(7i32, Nullability::NonNullable)),
            ("b", b(3)),
        ])
    };
    let core = variant_rows([
        Scalar::variant(variant_object([("b", b(1))])),
        Scalar::variant(variant_object([("b", b(2))])),
        Scalar::variant(third_row),
        Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
    ])?;

    let c = |value: i64| {
        Scalar::variant(variant_object([(
            "c",
            Scalar::primitive(value, Nullability::NonNullable),
        )]))
    };
    let fields = StructArray::try_new(
        FieldNames::from_iter(["a", "n"]),
        vec![
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, None]).into_array(),
            variant_rows([
                c(1),
                c(2),
                c(3),
                Scalar::null(DType::Variant(Nullability::Nullable)),
            ])?,
        ],
        4,
        Validity::from_iter([true, true, true, false]),
    )?
    .into_array();
    Ok((core, fields))
}

/// A test function over Variant data that fails when it is executed, which shows that a rule
/// does not read its input.
#[derive(Clone)]
struct Unreadable;

impl ScalarFnVTable for Unreadable {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.unreadable");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(arg_dtypes[0].clone())
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("the core storage must not be read")
    }
}

/// Each rule of [`VariantGet::reduce`] over a [`VariantPack`] extracts what the Variant that
/// the pack builds gives, and reads only the child that the rule says it reads.
#[rstest]
#[case::shredded_field_of_complete_pack(&["a"], nullable(PType::I32), true, Reduced::ToShredded)]
#[case::shredded_field_of_partial_pack(&["a"], nullable(PType::I32), false, Reduced::No)]
#[case::field_not_shredded(&["b"], nullable(PType::I32), true, Reduced::ToCore)]
#[case::path_into_shredded_variant(&["n", "c"], nullable(PType::I64), true, Reduced::ToShredded)]
#[case::leaf_of_other_type(&["a"], nullable(PType::I64), true, Reduced::No)]
#[case::struct_of_shredded_dtype(
    &[],
    shredded_struct(Nullability::Nullable),
    true,
    Reduced::ToShredded
)]
// The shredded `a` is null in the third row, which `variant_get` reads as a null row.
#[case::struct_with_non_nullable_field(
    &[],
    shredded_struct(Nullability::NonNullable),
    true,
    Reduced::No
)]
fn variant_get_over_variant_pack_reduces(
    #[case] path: &[&str],
    #[case] dtype: DType,
    #[case] complete: bool,
    #[case] reduced: Reduced,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (core, fields) = pack_over_core(complete)?;
    let packed_core = match reduced {
        Reduced::ToShredded => {
            let core = core
                .clone()
                .cast(DType::Variant(Nullability::NonNullable))?;
            ScalarFnArray::try_new(Unreadable.bind(EmptyOptions), [core])?.into_array()
        }
        Reduced::ToCore | Reduced::No => core.clone(),
    };
    let packed = VariantPack::try_new(packed_core.clone(), fields.clone(), complete)?.into_array();

    let expr = variant_get(
        root(),
        VariantPath::new(path.iter().map(|field| VariantPathElement::field(*field))),
        Some(dtype),
    );
    let actual = packed.apply(&expr)?.optimize_ctx(&ctx)?;

    let nodes = actual.depth_first_traversal().collect::<Vec<_>>();
    let has_pack = nodes.iter().any(|node| {
        node.as_opt::<ScalarFn>()
            .is_some_and(|scalar_fn| scalar_fn.scalar_fn().id() == VariantPack.id())
    });
    // The optimizer can push the extraction into the chunks of `core`, so any of its arrays
    // counts as reading it.
    let core_nodes = packed_core.depth_first_traversal().collect::<Vec<_>>();
    let reads_core = nodes.iter().any(|node| {
        core_nodes
            .iter()
            .any(|core_node| ArrayRef::ptr_eq(node, core_node))
    });
    match reduced {
        Reduced::ToShredded => assert!(!has_pack && !reads_core, "{actual}"),
        Reduced::ToCore => assert!(!has_pack && reads_core, "{actual}"),
        Reduced::No => assert!(has_pack, "{actual}"),
    }

    let expected = VariantArray::try_new(core, Some(fields))?
        .into_array()
        .apply(&expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    assert_arrays_eq!(actual.execute::<ArrayRef>(&mut ctx)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn complete_pack_respects_both_validities() -> VortexResult<()> {
    let core = variant_rows([
        Scalar::null(DType::Variant(Nullability::Nullable)),
        Scalar::variant(variant_object([])),
        Scalar::variant(variant_object([])),
    ])?;
    let fields = StructArray::from_fields(&[("a", buffer![7i32, 8, 9].into_array())])?.into_array();
    let validity = BoolArray::from_iter([true, false, true]).into_array();
    let packed = VariantPack::try_new(core, fields.mask(validity)?, true)?.into_array();
    let expr = variant_get(root(), VariantPath::field("a"), Some(nullable(PType::I32)));
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        packed
            .apply(&expr)?
            .optimize_ctx(&ctx)?
            .execute::<ArrayRef>(&mut ctx)?,
        PrimitiveArray::from_option_iter([None, None, Some(9i32)]),
        &mut ctx
    );
    Ok(())
}

fn nullable(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::Nullable)
}

/// The struct that [`pack_over_core`] shreds, nullable, with field `a` of the given nullability.
fn shredded_struct(a: Nullability) -> DType {
    DType::struct_(
        [
            ("a", DType::Primitive(PType::I32, a)),
            ("n", DType::Variant(Nullability::Nullable)),
        ],
        Nullability::Nullable,
    )
}
