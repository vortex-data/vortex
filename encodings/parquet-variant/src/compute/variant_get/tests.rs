// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Int32Array;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_array::builder::BinaryViewBuilder;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType as ArrowDataType;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantBuilder;
use parquet_variant::VariantBuilderExt;
use parquet_variant::VariantPath;
use parquet_variant::VariantPathElement;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::VariantArrayBuilder;
use parquet_variant_compute::json_to_variant;
use rstest::fixture;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Variant;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::expr::variant_get;
use vortex_array::expr::variant_get_as;
use vortex_array::scalar_fn::fns::variant_get::VariantPath as VortexVariantPath;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantData;

#[fixture]
fn ctx() -> ExecutionCtx {
    LEGACY_SESSION.create_execution_ctx()
}

#[fixture]
fn object_array() -> ArrayRef {
    let mut builder = VariantArrayBuilder::new(3);
    builder
        .new_object()
        .with_field("a", 1i32)
        .with_field("b", "x")
        .finish();
    builder
        .new_object()
        .with_field("a", 2i32)
        .with_field("c", true)
        .finish();
    builder.new_object().with_field("b", "y").finish();
    ParquetVariantData::from_arrow_variant(&builder.build()).unwrap()
}

#[fixture]
fn nullable_object_array() -> ArrayRef {
    let mut builder = VariantArrayBuilder::new(3);
    builder.new_object().with_field("a", 10i32).finish();
    builder.new_object().with_field("a", 20i32).finish();
    builder.new_object().with_field("a", 30i32).finish();

    let inner = builder.build().into_inner();
    let null_struct = StructArray::try_new(
        inner.fields().clone(),
        inner.columns().to_vec(),
        Some(NullBuffer::from(vec![true, false, true])),
    )
    .unwrap();
    let arrow_variant = ArrowVariantArray::try_new(&null_struct).unwrap();
    ParquetVariantData::from_arrow_variant(&arrow_variant).unwrap()
}

fn apply_variant_get(
    array: &ArrayRef,
    path: impl Into<VortexVariantPath>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let expr = variant_get(path, root());
    let array = array.clone().apply(&expr)?;
    array.execute::<ArrayRef>(ctx)
}

fn vortex_to_arrow_variant(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowVariantArray> {
    let variant = array.as_::<Variant>();
    let parquet_variant = variant.core_storage().as_::<ParquetVariant>();
    parquet_variant.to_arrow(ctx)
}

fn assert_variant_arrays_match(expected: &ArrowVariantArray, actual: &ArrowVariantArray) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    assert_eq!(
        actual.inner().column_names(),
        expected.inner().column_names(),
        "column mismatch"
    );
    assert_eq!(actual.inner().nulls(), expected.inner().nulls());
    assert_eq!(
        actual.inner().fields().len(),
        expected.inner().fields().len()
    );

    for (expected, actual) in expected
        .inner()
        .fields()
        .iter()
        .zip(actual.inner().fields().iter())
    {
        assert_eq!(actual.name(), expected.name());
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.is_nullable(), expected.is_nullable());
    }
    for index in 0..expected.len() {
        let expected_is_null = expected.is_null(index);
        let actual_is_null = actual.is_null(index);

        assert_eq!(
            actual_is_null, expected_is_null,
            "row {index}: null mismatch (vortex={actual_is_null}, arrow={expected_is_null})"
        );

        if !expected_is_null {
            assert_eq!(
                actual.value(index),
                expected.value(index),
                "row {index}: value mismatch"
            );
        }
    }
}

fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
    let mut builder = BinaryViewBuilder::new();
    for value in values {
        builder.append_value(*value);
    }
    Arc::new(builder.finish())
}

fn arrow_variant_from_json_rows(
    json_rows: &[&str],
    validity: Option<&[bool]>,
) -> ArrowVariantArray {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|row| Some(*row)).collect::<Vec<_>>(),
    ));
    let base_variant = json_to_variant(&arrow_strings).unwrap();

    match validity {
        Some(validity) => {
            let inner = base_variant.into_inner();
            let null_struct = StructArray::try_new(
                inner.fields().clone(),
                inner.columns().to_vec(),
                Some(NullBuffer::from(validity.to_vec())),
            )
            .unwrap();
            ArrowVariantArray::try_new(&null_struct).unwrap()
        }
        None => base_variant,
    }
}

macro_rules! assert_variant_get_matches_arrow {
    (json_rows = $json_rows:expr,field = $field:expr,ctx = $ctx:expr $(,)?) => {
        assert_variant_get_matches_arrow!(
            arrow_variant = arrow_variant_from_json_rows($json_rows, None),
            path = $field,
            arrow_path = VariantPath::try_from($field).unwrap(),
            ctx = $ctx,
        )
    };
    (
        json_rows =
        $json_rows:expr,validity =
        $validity:expr,field =
        $field:expr,ctx =
        $ctx:expr $(,)?
    ) => {
        assert_variant_get_matches_arrow!(
            arrow_variant = arrow_variant_from_json_rows($json_rows, Some($validity)),
            path = $field,
            arrow_path = VariantPath::try_from($field).unwrap(),
            ctx = $ctx,
        )
    };
    (
        json_rows =
        $json_rows:expr,path =
        $path:expr,arrow_path =
        $arrow_path:expr,ctx =
        $ctx:expr $(,)?
    ) => {
        assert_variant_get_matches_arrow!(
            arrow_variant = arrow_variant_from_json_rows($json_rows, None),
            path = $path,
            arrow_path = $arrow_path,
            ctx = $ctx,
        )
    };
    (
        arrow_variant =
        $arrow_variant:expr,path =
        $path:expr,arrow_path =
        $arrow_path:expr,ctx =
        $ctx:expr $(,)?
    ) => {{
        let arrow_variant = $arrow_variant;
        let path = $path;
        let arrow_path = $arrow_path;
        let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
        let arrow_result = parquet_variant_compute::variant_get(
            &arrow_input,
            GetOptions::new_with_path(arrow_path),
        )
        .unwrap();
        let expected = ArrowVariantArray::try_new(
            arrow_result.as_any().downcast_ref::<StructArray>().unwrap(),
        )
        .unwrap();

        let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        let actual = apply_variant_get(&vortex_input, path, $ctx)?;
        let actual = vortex_to_arrow_variant(&actual, $ctx)?;

        assert_variant_arrays_match(&expected, &actual);
        Ok(())
    }};
}

macro_rules! assert_variant_get_typed_matches_arrow {
    (
        json_rows =
        $json_rows:expr,path =
        $path:expr,arrow_path =
        $arrow_path:expr,as_dtype =
        $as_dtype:expr,arrow_dtype =
        $arrow_dtype:expr,ctx =
        $ctx:expr $(,)?
    ) => {{
        let arrow_variant = arrow_variant_from_json_rows($json_rows, None);
        let path = $path;
        let arrow_path = $arrow_path;
        let as_dtype = $as_dtype;
        let arrow_result =
            parquet_variant_compute::variant_get(
                &arrow_variant.clone().into(),
                GetOptions::new_with_path(arrow_path).with_as_type(Some(FieldRef::new(
                    Field::new("result", $arrow_dtype, true),
                ))),
            )
            .unwrap();
        let expected = ArrayRef::from_arrow(arrow_result.as_ref(), true)?;

        let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        let expr = variant_get_as(path, as_dtype.clone(), root());
        let array = vortex_input.apply(&expr)?;
        let actual = array.execute::<ArrayRef>($ctx)?;

        assert_eq!(actual.dtype(), &as_dtype.as_nullable());
        assert_arrays_eq!(actual, expected);
        Ok(())
    }};
}

#[rstest]
#[case("some_field", &[r#"{"some_field": 1234}"#])]
#[case("a", &[r#"{"a": 1, "b": 2}"#, r#"{"a": "hello"}"#, r#"{"b": 99}"#])]
#[case("nested", &[r#"{"nested": {"x": 1, "y": 2}}"#])]
#[case("missing", &[r#"{"a": 1}"#, r#"{"b": 2}"#])]
#[case("x", &[r#"{"x": true}"#, r#"{"x": false}"#, r#"{"x": null}"#])]
#[case("arr", &[r#"{"arr": [1, 2, 3]}"#])]
#[case("s", &[r#"{"s": "hello world"}"#, r#"{"s": ""}"#])]
#[case("n", &[r#"{"n": 3.14}"#, r#"{"n": -0.0}"#])]
fn test_variant_get_matches_arrow(
    #[case] field: &str,
    #[case] json_rows: &[&str],
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(json_rows = json_rows, field = field, ctx = &mut ctx)
}

#[rstest]
fn test_variant_get_matches_arrow_non_object(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(
        json_rows = &["42", r#""hello""#, "true", "null"],
        field = "a",
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_mixed_types(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(
        json_rows = &[
            r#"{"v": 1}"#,
            r#"{"v": "text"}"#,
            r#"{"v": true}"#,
            r#"{"v": [1,2]}"#,
            r#"{"v": {"nested": 1}}"#,
        ],
        field = "v",
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_nullable(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(
        json_rows = &[r#"{"a": 10}"#, r#"{"a": 20}"#, r#"{"a": 30}"#],
        validity = &[true, false, true],
        field = "a",
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_all_null(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(
        json_rows = &[r#"{"a": 1}"#, r#"{"a": 2}"#, r#"{"a": 3}"#],
        validity = &[false, false, false],
        field = "a",
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_nested_object_result(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_matches_arrow!(
        json_rows = &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"a": 1, "b": 2}}"#,
        ],
        field = "outer",
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_nested_path(mut ctx: ExecutionCtx) -> VortexResult<()> {
    let path = VortexVariantPath::from_name("outer").join("inner");
    let arrow_path = VariantPath::from_iter([
        VariantPathElement::field("outer".to_string()),
        VariantPathElement::field("inner".to_string()),
    ]);
    assert_variant_get_matches_arrow!(
        json_rows = &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"inner": "x"}}"#,
            r#"{"outer": {"other": true}}"#,
        ],
        path = path,
        arrow_path = arrow_path,
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_index_path(mut ctx: ExecutionCtx) -> VortexResult<()> {
    let path = VortexVariantPath::from_name("arr").join(1usize);
    let arrow_path = VariantPath::from_iter([
        VariantPathElement::field("arr".to_string()),
        VariantPathElement::index(1),
    ]);
    assert_variant_get_matches_arrow!(
        json_rows = &[
            r#"{"arr": [1, 2, 3]}"#,
            r#"{"arr": ["a", "b"]}"#,
            r#"{"arr": [true]}"#,
        ],
        path = path,
        arrow_path = arrow_path,
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_matches_arrow_typed_path(mut ctx: ExecutionCtx) -> VortexResult<()> {
    assert_variant_get_typed_matches_arrow!(
        json_rows = &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"inner": "x"}}"#,
            r#"{"outer": {"other": true}}"#,
        ],
        path = VortexVariantPath::from_name("outer").join("inner"),
        arrow_path = VariantPath::from_iter([
            VariantPathElement::field("outer".to_string()),
            VariantPathElement::field("inner".to_string()),
        ]),
        as_dtype = DType::Primitive(PType::I64, Nullability::NonNullable),
        arrow_dtype = ArrowDataType::Int64,
        ctx = &mut ctx,
    )
}

#[rstest]
fn test_variant_get_basic(object_array: ArrayRef, mut ctx: ExecutionCtx) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "a", &mut ctx)?;

    assert_eq!(result.len(), 3);

    let row0 = result.execute_scalar(0, &mut ctx)?;
    assert!(!row0.is_null());
    assert_eq!(*row0.as_variant().value().unwrap(), 1i32.into());

    let row1 = result.execute_scalar(1, &mut ctx)?;
    assert!(!row1.is_null());
    assert_eq!(*row1.as_variant().value().unwrap(), 2i32.into());

    assert!(result.execute_scalar(2, &mut ctx)?.is_null());
    Ok(())
}

#[rstest]
fn test_variant_get_missing_field(
    object_array: ArrayRef,
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "nonexistent", &mut ctx)?;

    assert_eq!(result.len(), 3);
    for index in 0..result.len() {
        assert!(
            result.execute_scalar(index, &mut ctx)?.is_null(),
            "row {index} should be null"
        );
    }

    Ok(())
}

#[rstest]
fn test_variant_get_null_input(
    nullable_object_array: ArrayRef,
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let result = apply_variant_get(&nullable_object_array, "a", &mut ctx)?;

    assert_eq!(result.len(), 3);
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(2, &mut ctx)?.is_null());

    Ok(())
}

#[rstest]
fn test_variant_get_non_object(mut ctx: ExecutionCtx) -> VortexResult<()> {
    let mut builder = VariantArrayBuilder::new(2);
    builder.append_variant(PqVariant::from(42i32));
    builder.append_variant(PqVariant::from("hello"));
    let array = ParquetVariantData::from_arrow_variant(&builder.build())?;

    let result = apply_variant_get(&array, "a", &mut ctx)?;

    assert_eq!(result.len(), 2);
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());

    Ok(())
}

#[rstest]
fn test_variant_get_different_field(
    object_array: ArrayRef,
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "b", &mut ctx)?;

    assert_eq!(result.len(), 3);
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(2, &mut ctx)?.is_null());

    Ok(())
}

#[rstest]
fn test_variant_get_through_slice_wrapper(
    object_array: ArrayRef,
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let expr = variant_get("a", root());
    let actual = object_array
        .slice(1..3)?
        .apply(&expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    let expected = apply_variant_get(&object_array, "a", &mut ctx)?;

    assert_eq!(actual.len(), 2);
    assert!(
        actual
            .as_::<Variant>()
            .core_storage()
            .is::<ParquetVariant>()
    );
    let actual_row0 = actual.execute_scalar(0, &mut ctx)?;
    let expected_row1 = expected.execute_scalar(1, &mut ctx)?;
    assert_eq!(actual_row0, expected_row1);
    let actual_row1 = actual.execute_scalar(1, &mut ctx)?;
    let expected_row2 = expected.execute_scalar(2, &mut ctx)?;
    assert_eq!(actual_row1, expected_row2);
    Ok(())
}

#[rstest]
fn test_variant_get_through_filter_wrapper(
    object_array: ArrayRef,
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let mask = Mask::from_iter([true, false, true]);
    let expr = variant_get("a", root());
    let actual = object_array
        .filter(mask)?
        .apply(&expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    let expected = apply_variant_get(&object_array, "a", &mut ctx)?;

    assert_eq!(actual.len(), 2);
    assert!(
        actual
            .as_::<Variant>()
            .core_storage()
            .is::<ParquetVariant>()
    );
    let actual_row0 = actual.execute_scalar(0, &mut ctx)?;
    let expected_row0 = expected.execute_scalar(0, &mut ctx)?;
    assert_eq!(actual_row0, expected_row0);
    let actual_row1 = actual.execute_scalar(1, &mut ctx)?;
    let expected_row2 = expected.execute_scalar(2, &mut ctx)?;
    assert_eq!(actual_row1, expected_row2);
    Ok(())
}

#[rstest]
fn variant_get_on_canonical_variant_preserves_parquet_core_and_shredded_child(
    mut ctx: ExecutionCtx,
) -> VortexResult<()> {
    let mut builder = VariantBuilder::new();
    builder
        .new_object()
        .with_field("a", 1i32)
        .with_field("b", "leftover")
        .finish();
    let (metadata, value) = builder.finish();

    let shredded_a = StructArray::try_new(
        vec![Arc::new(Field::new("typed_value", DataType::Int32, false))].into(),
        vec![Arc::new(Int32Array::from(vec![7]))],
        None,
    )?;
    let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
        vec![Arc::new(Field::new(
            "a",
            shredded_a.data_type().clone(),
            false,
        ))]
        .into(),
        vec![Arc::new(shredded_a)],
        None,
    )?);

    let struct_array = StructArray::try_new(
        vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, false)),
            Arc::new(Field::new("value", DataType::BinaryView, true)),
            Arc::new(Field::new(
                "typed_value",
                typed_value.data_type().clone(),
                false,
            )),
        ]
        .into(),
        vec![
            binary_view_array(&[metadata.as_slice()]),
            binary_view_array(&[value.as_slice()]),
            typed_value,
        ],
        None,
    )?;
    let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
    let expected_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
    let expected = parquet_variant_compute::variant_get(
        &expected_input,
        GetOptions::new_with_path(VariantPath::try_from("a")?),
    )?;
    let expected =
        ArrowVariantArray::try_new(expected.as_any().downcast_ref::<StructArray>().unwrap())?;

    let canonical_variant = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let result = apply_variant_get(&canonical_variant, "a", &mut ctx)?;
    let result_variant = result.as_::<Variant>();

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    assert!(result_variant.core_storage().is::<ParquetVariant>());
    assert!(result_variant.shredded().is_some());

    let actual = vortex_to_arrow_variant(&result, &mut ctx)?;
    assert_variant_arrays_match(&expected, &actual);
    assert_eq!(actual.value(0), expected.value(0));

    Ok(())
}

#[rstest]
fn variant_get_missing_path_returns_nullable_nulls(mut ctx: ExecutionCtx) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"a": 1}"#),
        Some(r#"{"b": 2}"#),
    ]));
    let arrow_variant = json_to_variant(&arrow_strings)?;
    let canonical_variant = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let result = apply_variant_get(&canonical_variant, "missing", &mut ctx)?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());

    Ok(())
}
