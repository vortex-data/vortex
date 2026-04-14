// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType as ArrowDataType;
use arrow_schema::Field as ArrowField;
use arrow_schema::FieldRef;
use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantBuilderExt;
use parquet_variant::VariantPath;
use parquet_variant::VariantPathElement;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::VariantArrayBuilder;
use parquet_variant_compute::json_to_variant;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
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

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantData;

/// Apply variant_get and execute through the full pipeline (including execute_parent).
fn apply_variant_get(arr: &ArrayRef, path: impl Into<VortexVariantPath>) -> VortexResult<ArrayRef> {
    let expr = variant_get(path, root());
    let array = arr.clone().apply(&expr)?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array.execute::<ArrayRef>(&mut ctx)
}

/// Apply typed variant_get and execute through the full pipeline.
fn apply_variant_get_as(
    arr: &ArrayRef,
    path: impl Into<VortexVariantPath>,
    as_dtype: DType,
) -> VortexResult<ArrayRef> {
    let expr = variant_get_as(path, as_dtype, root());
    let array = arr.clone().apply(&expr)?;
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array.execute::<ArrayRef>(&mut ctx)
}

/// Convert a Vortex result back to an Arrow VariantArray for comparison.
fn vortex_to_arrow_variant(arr: &ArrayRef) -> ArrowVariantArray {
    let variant = arr.as_::<vortex_array::arrays::Variant>();
    let pv = variant.child().as_::<ParquetVariant>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    pv.to_arrow(&mut ctx).unwrap()
}

/// Run variant_get through both Arrow and Vortex on the same input, and assert
/// the per-row results (value + validity) are identical by comparing at the Arrow level.
fn assert_matches_arrow(json_rows: &[&str], field: &str) {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let path = VariantPath::try_from(field).unwrap();
    let arrow_result = parquet_variant_compute::variant_get(
        &arrow_variant.clone().into(),
        GetOptions::new_with_path(path),
    )
    .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant).unwrap();
    let vortex_result = apply_variant_get(&vortex_input, field).unwrap();
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result);

    assert_eq!(
        vortex_as_arrow.len(),
        arrow_result_variant.len(),
        "length mismatch"
    );

    for i in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(i);
        let vortex_is_null = vortex_as_arrow.is_null(i);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {i}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            let arrow_value = arrow_result_variant.value(i);
            let vortex_value = vortex_as_arrow.value(i);
            assert_eq!(
                vortex_value, arrow_value,
                "row {i}: value mismatch\n  vortex: {vortex_value:?}\n  arrow:  {arrow_value:?}"
            );
        }
    }
}

/// Run variant_get through both Arrow and Vortex for an explicit nested/index path,
/// and assert the per-row results match at the Arrow variant level.
fn assert_matches_arrow_with_path(
    json_rows: &[&str],
    path: VortexVariantPath,
    arrow_path: VariantPath<'static>,
) {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let arrow_result = parquet_variant_compute::variant_get(
        &arrow_variant.clone().into(),
        GetOptions::new_with_path(arrow_path),
    )
    .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant).unwrap();
    let vortex_result = apply_variant_get(&vortex_input, path).unwrap();
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result);

    assert_eq!(
        vortex_as_arrow.len(),
        arrow_result_variant.len(),
        "length mismatch"
    );

    for i in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(i);
        let vortex_is_null = vortex_as_arrow.is_null(i);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {i}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            let arrow_value = arrow_result_variant.value(i);
            let vortex_value = vortex_as_arrow.value(i);
            assert_eq!(
                vortex_value, arrow_value,
                "row {i}: value mismatch\n  vortex: {vortex_value:?}\n  arrow:  {arrow_value:?}"
            );
        }
    }
}

/// Run typed variant_get through both Arrow and Vortex for an explicit path,
/// and assert the typed nullable result matches.
fn assert_typed_matches_arrow_with_path(
    json_rows: &[&str],
    path: VortexVariantPath,
    arrow_path: VariantPath<'static>,
    as_dtype: DType,
    arrow_dtype: ArrowDataType,
) -> VortexResult<()> {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
    ));
    let arrow_variant = json_to_variant(&arrow_strings).unwrap();
    let arrow_result =
        parquet_variant_compute::variant_get(
            &arrow_variant.clone().into(),
            GetOptions::new_with_path(arrow_path).with_as_type(Some(FieldRef::new(
                ArrowField::new("result", arrow_dtype, true),
            ))),
        )
        .unwrap();
    let expected = ArrayRef::from_arrow(arrow_result.as_ref(), true)?;

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant).unwrap();
    let vortex_result = apply_variant_get_as(&vortex_input, path, as_dtype.clone())?;

    assert_eq!(vortex_result.dtype(), &as_dtype.as_nullable());
    assert_arrays_eq!(vortex_result, expected);
    Ok(())
}

/// Run variant_get through both Arrow and Vortex on nullable input (with NullBuffer),
/// and assert the results match.
fn assert_matches_arrow_nullable(json_rows: &[&str], validity: &[bool], field: &str) {
    let arrow_strings: ArrowArrayRef = Arc::new(StringArray::from(
        json_rows.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
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

    let path = VariantPath::try_from(field).unwrap();
    let arrow_result = parquet_variant_compute::variant_get(
        &ArrowArrayRef::from(arrow_variant.clone()),
        GetOptions::new_with_path(path),
    )
    .unwrap();
    let arrow_result_variant =
        ArrowVariantArray::try_new(arrow_result.as_any().downcast_ref::<StructArray>().unwrap())
            .unwrap();

    let vortex_input = ParquetVariantData::from_arrow_variant(&arrow_variant).unwrap();
    let vortex_result = apply_variant_get(&vortex_input, field).unwrap();
    let vortex_as_arrow = vortex_to_arrow_variant(&vortex_result);

    assert_eq!(
        vortex_as_arrow.len(),
        arrow_result_variant.len(),
        "length mismatch"
    );

    for i in 0..arrow_result_variant.len() {
        let arrow_is_null = arrow_result_variant.is_null(i);
        let vortex_is_null = vortex_as_arrow.is_null(i);

        assert_eq!(
            vortex_is_null, arrow_is_null,
            "row {i}: null mismatch (vortex={vortex_is_null}, arrow={arrow_is_null})"
        );

        if !arrow_is_null {
            let arrow_value = arrow_result_variant.value(i);
            let vortex_value = vortex_as_arrow.value(i);
            assert_eq!(
                vortex_value, arrow_value,
                "row {i}: value mismatch\n  vortex: {vortex_value:?}\n  arrow:  {arrow_value:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests that compare Vortex vs Arrow variant_get
// ---------------------------------------------------------------------------

#[rstest]
#[case("some_field", &[r#"{"some_field": 1234}"#])]
#[case("a", &[r#"{"a": 1, "b": 2}"#, r#"{"a": "hello"}"#, r#"{"b": 99}"#])]
#[case("nested", &[r#"{"nested": {"x": 1, "y": 2}}"#])]
#[case("missing", &[r#"{"a": 1}"#, r#"{"b": 2}"#])]
#[case("x", &[r#"{"x": true}"#, r#"{"x": false}"#, r#"{"x": null}"#])]
#[case("arr", &[r#"{"arr": [1, 2, 3]}"#])]
#[case("s", &[r#"{"s": "hello world"}"#, r#"{"s": ""}"#])]
#[case("n", &[r#"{"n": 3.14}"#, r#"{"n": -0.0}"#])]
fn test_variant_get_matches_arrow(#[case] field: &str, #[case] json_rows: &[&str]) {
    assert_matches_arrow(json_rows, field);
}

#[test]
fn test_variant_get_matches_arrow_non_object() {
    // Primitive variants (not objects) — accessing any field should give null
    assert_matches_arrow(&["42", r#""hello""#, "true", "null"], "a");
}

#[test]
fn test_variant_get_matches_arrow_mixed_types() {
    // Same field name, different value types across rows
    assert_matches_arrow(
        &[
            r#"{"v": 1}"#,
            r#"{"v": "text"}"#,
            r#"{"v": true}"#,
            r#"{"v": [1,2]}"#,
            r#"{"v": {"nested": 1}}"#,
        ],
        "v",
    );
}

#[test]
fn test_variant_get_matches_arrow_nullable() {
    assert_matches_arrow_nullable(
        &[r#"{"a": 10}"#, r#"{"a": 20}"#, r#"{"a": 30}"#],
        &[true, false, true], // row 1 is null
        "a",
    );
}

#[test]
fn test_variant_get_matches_arrow_all_null() {
    assert_matches_arrow_nullable(
        &[r#"{"a": 1}"#, r#"{"a": 2}"#, r#"{"a": 3}"#],
        &[false, false, false],
        "a",
    );
}

#[test]
fn test_variant_get_matches_arrow_nested_object_result() {
    // The result of variant_get is itself an object
    assert_matches_arrow(
        &[
            r#"{"outer": {"inner": 42}}"#,
            r#"{"outer": {"a": 1, "b": 2}}"#,
        ],
        "outer",
    );
}

#[test]
fn test_variant_get_matches_arrow_nested_path() {
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
    );
}

#[test]
fn test_variant_get_matches_arrow_index_path() {
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
    );
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

// ---------------------------------------------------------------------------
// Original standalone tests
// ---------------------------------------------------------------------------

#[test]
fn test_variant_get_basic() -> VortexResult<()> {
    let arr = make_object_array()?;
    let result = apply_variant_get(&arr, "a")?;

    assert_eq!(result.len(), 3);

    // Row 0: {"a": 1, ...} → variant(1)
    let s0 = result.scalar_at(0)?;
    assert!(!s0.is_null());
    let inner0 = s0.as_variant().value().unwrap();
    assert_eq!(*inner0, 1i32.into());

    // Row 1: {"a": 2, ...} → variant(2)
    let s1 = result.scalar_at(1)?;
    assert!(!s1.is_null());
    let inner1 = s1.as_variant().value().unwrap();
    assert_eq!(*inner1, 2i32.into());

    // Row 2: {"b": "y"} → null (field "a" missing)
    let s2 = result.scalar_at(2)?;
    assert!(s2.is_null());

    Ok(())
}

#[test]
fn test_variant_get_missing_field() -> VortexResult<()> {
    let arr = make_object_array()?;
    let result = apply_variant_get(&arr, "nonexistent")?;

    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert!(result.scalar_at(i)?.is_null(), "row {i} should be null");
    }

    Ok(())
}

#[test]
fn test_variant_get_null_input() -> VortexResult<()> {
    let arr = make_nullable_object_array()?;
    let result = apply_variant_get(&arr, "a")?;

    assert_eq!(result.len(), 3);
    assert!(!result.scalar_at(0)?.is_null());
    assert!(result.scalar_at(1)?.is_null());
    assert!(!result.scalar_at(2)?.is_null());

    Ok(())
}

#[test]
fn test_variant_get_non_object() -> VortexResult<()> {
    let mut builder = VariantArrayBuilder::new(2);
    builder.append_variant(PqVariant::from(42i32));
    builder.append_variant(PqVariant::from("hello"));
    let arr = ParquetVariantData::from_arrow_variant(&builder.build())?;

    let result = apply_variant_get(&arr, "a")?;

    assert_eq!(result.len(), 2);
    assert!(result.scalar_at(0)?.is_null());
    assert!(result.scalar_at(1)?.is_null());

    Ok(())
}

#[test]
fn test_variant_get_different_field() -> VortexResult<()> {
    let arr = make_object_array()?;
    let result = apply_variant_get(&arr, "b")?;

    assert_eq!(result.len(), 3);
    assert!(!result.scalar_at(0)?.is_null());
    assert!(result.scalar_at(1)?.is_null());
    assert!(!result.scalar_at(2)?.is_null());

    Ok(())
}

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

/// Build a small non-null object variant array used by the standalone tests.
fn make_object_array() -> VortexResult<ArrayRef> {
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

    ParquetVariantData::from_arrow_variant(&builder.build())
}

/// Build the same object array shape with an explicit top-level validity bitmap.
fn make_nullable_object_array() -> VortexResult<ArrayRef> {
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
    ParquetVariantData::from_arrow_variant(&arrow_variant)
}
