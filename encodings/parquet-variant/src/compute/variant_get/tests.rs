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

fn apply_variant_get(
    array: &ArrayRef,
    path: impl Into<VortexVariantPath>,
) -> VortexResult<ArrayRef> {
    let expr = variant_get(path, root());
    let array = array.clone().apply(&expr)?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array.execute::<ArrayRef>(&mut ctx)
}

fn vortex_to_arrow_variant(array: &ArrayRef) -> VortexResult<ArrowVariantArray> {
    let variant = array.as_::<Variant>();
    let parquet_variant = variant.core_storage().as_::<ParquetVariant>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    parquet_variant.to_arrow(&mut ctx)
}

fn assert_variant_storage_matches(expected: &ArrowVariantArray, actual: &ArrowVariantArray) {
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
}

fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
    let mut builder = BinaryViewBuilder::new();
    for value in values {
        builder.append_value(*value);
    }
    Arc::new(builder.finish())
}

fn assert_matches_arrow(json_rows: &[&str], field: &str) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|row| Some(*row)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
    let arrow_result = parquet_variant_compute::variant_get(
        &arrow_input,
        GetOptions::new_with_path(VariantPath::try_from(field).unwrap()),
    )
    .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let vortex_result = apply_variant_get(&vortex_input, field)?;
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result)?;

    assert_variant_storage_matches(&arrow_result_variant, &vortex_as_arrow);

    for index in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(index);
        let vortex_is_null = vortex_as_arrow.is_null(index);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {index}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            assert_eq!(
                vortex_as_arrow.value(index),
                arrow_result_variant.value(index),
                "row {index}: value mismatch"
            );
        }
    }

    Ok(())
}

fn assert_matches_arrow_with_path(
    json_rows: &[&str],
    path: VortexVariantPath,
    arrow_path: VariantPath<'static>,
) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|row| Some(*row)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
    let arrow_result =
        parquet_variant_compute::variant_get(&arrow_input, GetOptions::new_with_path(arrow_path))
            .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let vortex_result = apply_variant_get(&vortex_input, path)?;
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result)?;

    assert_variant_storage_matches(&arrow_result_variant, &vortex_as_arrow);

    for index in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(index);
        let vortex_is_null = vortex_as_arrow.is_null(index);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {index}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            assert_eq!(
                vortex_as_arrow.value(index),
                arrow_result_variant.value(index),
                "row {index}: value mismatch"
            );
        }
    }

    Ok(())
}

fn assert_typed_matches_arrow_with_path(
    json_rows: &[&str],
    path: VortexVariantPath,
    arrow_path: VariantPath<'static>,
    as_dtype: DType,
    arrow_dtype: ArrowDataType,
) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|row| Some(*row)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let arrow_result = parquet_variant_compute::variant_get(
        &arrow_variant.clone().into(),
        GetOptions::new_with_path(arrow_path).with_as_type(Some(FieldRef::new(Field::new(
            "result",
            arrow_dtype,
            true,
        )))),
    )
    .unwrap();
    let expected = ArrayRef::from_arrow(arrow_result.as_ref(), true)?;

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let expr = variant_get_as(path, as_dtype.clone(), root());
    let array = vortex_input.apply(&expr)?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let actual = array.execute::<ArrayRef>(&mut ctx)?;

    assert_eq!(actual.dtype(), &as_dtype.as_nullable());
    assert_arrays_eq!(actual, expected);
    Ok(())
}

fn assert_matches_arrow_nullable(
    json_rows: &[&str],
    validity: &[bool],
    field: &str,
) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|row| Some(*row)).collect::<Vec<_>>(),
    ));
    let base_variant = json_to_variant(&arrow_strings).unwrap();
    let inner = base_variant.into_inner();
    let null_struct = StructArray::try_new(
        inner.fields().clone(),
        inner.columns().to_vec(),
        Some(NullBuffer::from(validity.to_vec())),
    )
    .unwrap();
    let arrow_variant = ArrowVariantArray::try_new(&null_struct).unwrap();
    let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
    let arrow_result = parquet_variant_compute::variant_get(
        &arrow_input,
        GetOptions::new_with_path(VariantPath::try_from(field).unwrap()),
    )
    .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let vortex_result = apply_variant_get(&vortex_input, field)?;
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result)?;

    assert_variant_storage_matches(&arrow_result_variant, &vortex_as_arrow);

    for index in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(index);
        let vortex_is_null = vortex_as_arrow.is_null(index);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {index}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            assert_eq!(
                vortex_as_arrow.value(index),
                arrow_result_variant.value(index),
                "row {index}: value mismatch"
            );
        }
    }

    Ok(())
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
) -> VortexResult<()> {
    assert_matches_arrow(json_rows, field)
}

#[test]
fn test_variant_get_matches_arrow_non_object() -> VortexResult<()> {
    assert_matches_arrow(&["42", r#""hello""#, "true", "null"], "a")
}

#[test]
fn test_variant_get_matches_arrow_mixed_types() -> VortexResult<()> {
    assert_matches_arrow(
        &[
            r#"{"v": 1}"#,
            r#"{"v": "text"}"#,
            r#"{"v": true}"#,
            r#"{"v": [1,2]}"#,
            r#"{"v": {"nested": 1}}"#,
        ],
        "v",
    )
}

#[test]
fn test_variant_get_matches_arrow_nullable() -> VortexResult<()> {
    assert_matches_arrow_nullable(
        &[r#"{"a": 10}"#, r#"{"a": 20}"#, r#"{"a": 30}"#],
        &[true, false, true],
        "a",
    )
}

#[test]
fn test_variant_get_matches_arrow_all_null() -> VortexResult<()> {
    assert_matches_arrow_nullable(
        &[r#"{"a": 1}"#, r#"{"a": 2}"#, r#"{"a": 3}"#],
        &[false, false, false],
        "a",
    )
}

#[test]
fn test_variant_get_matches_arrow_nested_object_result() -> VortexResult<()> {
    assert_matches_arrow(
        &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"a": 1, "b": 2}}"#,
        ],
        "outer",
    )
}

#[test]
fn test_variant_get_matches_arrow_nested_path() -> VortexResult<()> {
    let path = VortexVariantPath::from_name("outer").join("inner");
    let arrow_path = VariantPath::from_iter([
        VariantPathElement::field("outer".to_string()),
        VariantPathElement::field("inner".to_string()),
    ]);
    assert_matches_arrow_with_path(
        &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"inner": "x"}}"#,
            r#"{"outer": {"other": true}}"#,
        ],
        path,
        arrow_path,
    )
}

#[test]
fn test_variant_get_matches_arrow_index_path() -> VortexResult<()> {
    let path = VortexVariantPath::from_name("arr").join(1usize);
    let arrow_path = VariantPath::from_iter([
        VariantPathElement::field("arr".to_string()),
        VariantPathElement::index(1),
    ]);
    assert_matches_arrow_with_path(
        &[
            r#"{"arr": [1, 2, 3]}"#,
            r#"{"arr": ["a", "b"]}"#,
            r#"{"arr": [true]}"#,
        ],
        path,
        arrow_path,
    )
}

#[test]
fn test_variant_get_matches_arrow_typed_path() -> VortexResult<()> {
    assert_typed_matches_arrow_with_path(
        &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"inner": "x"}}"#,
            r#"{"outer": {"other": true}}"#,
        ],
        VortexVariantPath::from_name("outer").join("inner"),
        VariantPath::from_iter([
            VariantPathElement::field("outer".to_string()),
            VariantPathElement::field("inner".to_string()),
        ]),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        ArrowDataType::Int64,
    )
}

#[rstest]
fn test_variant_get_basic(object_array: ArrayRef) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "a")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

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
fn test_variant_get_missing_field(object_array: ArrayRef) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "nonexistent")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

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
fn test_variant_get_null_input(nullable_object_array: ArrayRef) -> VortexResult<()> {
    let result = apply_variant_get(&nullable_object_array, "a")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    assert_eq!(result.len(), 3);
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(2, &mut ctx)?.is_null());

    Ok(())
}

#[test]
fn test_variant_get_non_object() -> VortexResult<()> {
    let mut builder = VariantArrayBuilder::new(2);
    builder.append_variant(PqVariant::from(42i32));
    builder.append_variant(PqVariant::from("hello"));
    let array = ParquetVariantData::from_arrow_variant(&builder.build())?;

    let result = apply_variant_get(&array, "a")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    assert_eq!(result.len(), 2);
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());

    Ok(())
}

#[rstest]
fn test_variant_get_different_field(object_array: ArrayRef) -> VortexResult<()> {
    let result = apply_variant_get(&object_array, "b")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    assert_eq!(result.len(), 3);
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());
    assert!(!result.execute_scalar(2, &mut ctx)?.is_null());

    Ok(())
}

#[rstest]
fn test_variant_get_through_slice_wrapper(object_array: ArrayRef) -> VortexResult<()> {
    let expr = variant_get("a", root());
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let actual = object_array
        .slice(1..3)?
        .apply(&expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    let expected = apply_variant_get(&object_array, "a")?;

    assert_eq!(actual.len(), 2);
    assert!(
        actual
            .as_::<Variant>()
            .core_storage()
            .is::<ParquetVariant>()
    );
    let mut actual_ctx = LEGACY_SESSION.create_execution_ctx();
    let mut expected_ctx = LEGACY_SESSION.create_execution_ctx();
    assert_eq!(
        actual.execute_scalar(0, &mut actual_ctx)?,
        expected.execute_scalar(1, &mut expected_ctx)?
    );
    assert_eq!(
        actual.execute_scalar(1, &mut actual_ctx)?,
        expected.execute_scalar(2, &mut expected_ctx)?
    );
    Ok(())
}

#[rstest]
fn test_variant_get_through_filter_wrapper(object_array: ArrayRef) -> VortexResult<()> {
    let mask = Mask::from_iter([true, false, true]);
    let expr = variant_get("a", root());
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let actual = object_array
        .filter(mask)?
        .apply(&expr)?
        .execute::<ArrayRef>(&mut ctx)?;
    let expected = apply_variant_get(&object_array, "a")?;

    assert_eq!(actual.len(), 2);
    assert!(
        actual
            .as_::<Variant>()
            .core_storage()
            .is::<ParquetVariant>()
    );
    let mut actual_ctx = LEGACY_SESSION.create_execution_ctx();
    let mut expected_ctx = LEGACY_SESSION.create_execution_ctx();
    assert_eq!(
        actual.execute_scalar(0, &mut actual_ctx)?,
        expected.execute_scalar(0, &mut expected_ctx)?
    );
    assert_eq!(
        actual.execute_scalar(1, &mut actual_ctx)?,
        expected.execute_scalar(2, &mut expected_ctx)?
    );
    Ok(())
}

#[test]
fn variant_get_on_canonical_variant_preserves_parquet_core_and_shredded_child() -> VortexResult<()>
{
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
    )
    .unwrap();
    let typed_value: ArrowArrayRef = Arc::new(
        StructArray::try_new(
            vec![Arc::new(Field::new(
                "a",
                shredded_a.data_type().clone(),
                false,
            ))]
            .into(),
            vec![Arc::new(shredded_a)],
            None,
        )
        .unwrap(),
    );

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
    )
    .unwrap();
    let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();
    let expected_input: ArrowArrayRef = Arc::new(arrow_variant.clone().into_inner());
    let expected = parquet_variant_compute::variant_get(
        &expected_input,
        GetOptions::new_with_path(VariantPath::try_from("a").unwrap()),
    )
    .unwrap();
    let expected =
        ArrowVariantArray::try_new(expected.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let canonical_variant = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let result = apply_variant_get(&canonical_variant, "a")?;
    let result_variant = result.as_::<Variant>();

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    assert!(result_variant.core_storage().is::<ParquetVariant>());
    assert!(result_variant.shredded().is_some());

    let actual = vortex_to_arrow_variant(&result)?;
    assert_variant_storage_matches(&expected, &actual);
    assert_eq!(actual.value(0), expected.value(0));

    Ok(())
}

#[test]
fn variant_get_missing_path_returns_nullable_nulls() -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"a": 1}"#),
        Some(r#"{"b": 2}"#),
    ]));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let canonical_variant = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
    let result = apply_variant_get(&canonical_variant, "missing")?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    assert!(result.execute_scalar(0, &mut ctx)?.is_null());
    assert!(result.execute_scalar(1, &mut ctx)?.is_null());

    Ok(())
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
