// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar_is_null;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::ScalarFnVTable;
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
use vortex_session::VortexSession;

use crate::expr::variant_get;
use crate::variant_get::VariantGet;
use crate::variant_get::VariantGetOptions;
use crate::variant_get::VariantPath;
use crate::variant_get::VariantPathElement;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();

    vortex_array::initialize(&session);

    crate::initialize(&session);

    session
});

fn variant_object(fields: impl IntoIterator<Item = (&'static str, Scalar)>) -> Scalar {
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

fn variant_rows(rows: impl IntoIterator<Item = Scalar>) -> VortexResult<ArrayRef> {
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

#[rstest]
#[case::list(false)]
#[case::fixed_size_list(true)]
fn variant_get_untyped_list_falls_back_to_core(#[case] fixed_size: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let row0_list = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    );
    let row1_list = Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(40i32, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    );
    let core_storage = variant_rows([
        Scalar::variant(row0_list.clone()),
        Scalar::variant(row1_list.clone()),
    ])?;
    let shredded_items = if fixed_size {
        FixedSizeListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?
        .into_array()
    } else {
        ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array()
    };
    let variant = VariantArray::try_new(core_storage, Some(shredded_items))?.into_array();

    let expr = variant_get(root(), VariantPath::root(), None);
    let result = variant.apply(&expr)?;

    assert_arrays_eq!(
        result,
        variant_rows([Scalar::variant(row0_list), Scalar::variant(row1_list)])?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn variant_get_dtype_mismatch_falls_back_to_core() -> VortexResult<()> {
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

#[test]
fn variant_get_non_nullable_nested_target_returns_null_without_panicking() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let row0 = Scalar::variant(Scalar::struct_(
        DType::Struct(
            StructFields::new(
                FieldNames::from(["x"]),
                vec![DType::Variant(Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        ),
        vec![Scalar::variant(Scalar::utf8(
            "not_an_int",
            Nullability::NonNullable,
        ))],
    ));
    let row1 = Scalar::variant(Scalar::struct_(
        DType::Struct(
            StructFields::new(
                FieldNames::from(["x"]),
                vec![DType::Variant(Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        ),
        vec![Scalar::variant(Scalar::primitive(
            7i32,
            Nullability::NonNullable,
        ))],
    ));
    let input = ChunkedArray::try_new(
        vec![
            ConstantArray::new(row0, 1).into_array(),
            ConstantArray::new(row1, 1).into_array(),
        ],
        DType::Variant(Nullability::NonNullable),
    )?
    .into_array();

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

    let list_row0 = Scalar::variant(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![Scalar::variant(Scalar::utf8(
            "bad",
            Nullability::NonNullable,
        ))],
        Nullability::NonNullable,
    ));
    let list_row1 = Scalar::variant(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![Scalar::variant(Scalar::primitive(
            9i32,
            Nullability::NonNullable,
        ))],
        Nullability::NonNullable,
    ));
    let list_input = ChunkedArray::try_new(
        vec![
            ConstantArray::new(list_row0, 1).into_array(),
            ConstantArray::new(list_row1, 1).into_array(),
        ],
        DType::Variant(Nullability::NonNullable),
    )?
    .into_array();

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
    let row0 = Scalar::variant(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(1i64, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(2i64, Nullability::NonNullable)),
        ],
        Nullability::NonNullable,
    ));
    let row1 = Scalar::variant(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![Scalar::variant(Scalar::primitive(
            3i64,
            Nullability::NonNullable,
        ))],
        Nullability::NonNullable,
    ));
    let input = ChunkedArray::try_new(
        vec![
            ConstantArray::new(row0, 1).into_array(),
            ConstantArray::new(row1, 1).into_array(),
        ],
        DType::Variant(Nullability::NonNullable),
    )?
    .into_array();

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
    let variant_null = Scalar::variant(Scalar::null(DType::Null));
    let row = Scalar::variant(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        vec![
            Scalar::variant(Scalar::primitive(7i32, Nullability::NonNullable)),
            variant_null,
        ],
        Nullability::NonNullable,
    ));
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
