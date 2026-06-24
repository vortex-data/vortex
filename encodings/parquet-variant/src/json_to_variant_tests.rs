// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution tests for the `vortex.json_to_variant` scalar function.
//!
//! The function definition lives in `vortex-json`; the JSON->Variant construction is performed
//! by the execute-parent kernel registered here, so these end-to-end tests live in
//! `vortex-parquet-variant` where that kernel is registered.

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar_is_null;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::expr::variant_get;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_json::Json;
use vortex_session::VortexSession;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ShreddingSpec;
use crate::json_to_variant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn i64_dtype() -> DType {
    DType::Primitive(PType::I64, Nullability::Nullable)
}

fn shred_field_as_i64(field: &str) -> VortexResult<ShreddingSpec> {
    ShreddingSpec::try_new([(VariantPath::field(field), i64_dtype())])
}

fn json_input(storage: ArrayRef) -> VortexResult<ArrayRef> {
    Ok(ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage)?.into_array())
}

fn execute_json_to_variant(input: ArrayRef, shredding: ShreddingSpec) -> VortexResult<ArrayRef> {
    let expr = json_to_variant(root(), shredding);
    input
        .apply(&expr)?
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
}

fn assert_variant_i64_rows(array: &ArrayRef, expected: &[Option<i64>]) -> VortexResult<()> {
    assert_eq!(array.len(), expected.len());
    let mut ctx = SESSION.create_execution_ctx();
    for (idx, expected) in expected.iter().enumerate() {
        let scalar = array.execute_scalar(idx, &mut ctx)?;
        let variant = scalar.as_variant();
        match expected {
            Some(expected) => {
                let value = variant
                    .value()
                    .ok_or_else(|| vortex_err!("expected non-null variant at row {idx}"))?
                    .cast(&i64_dtype())?;
                assert_eq!(value.as_primitive().typed_value::<i64>(), Some(*expected));
            }
            None => assert!(scalar.is_null(), "expected null row {idx}"),
        }
    }
    Ok(())
}

#[test]
fn rejects_bare_utf8_input() {
    let input = VarBinViewArray::from_iter_str(["1", "2"]).into_array();

    let err = execute_json_to_variant(input, ShreddingSpec::empty()).unwrap_err();
    assert!(
        err.to_string().contains("Json extension"),
        "unexpected error: {err}"
    );
}

#[test]
fn converts_json_extension_rows() -> VortexResult<()> {
    let input = json_input(
        VarBinViewArray::from_iter_str([r#"{"a": 1}"#, "2", r#"{"a": 3}"#]).into_array(),
    )?;

    let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::NonNullable));
    let mut ctx = SESSION.create_execution_ctx();
    let row0 = result.execute_scalar(0, &mut ctx)?;
    let object = row0
        .as_variant()
        .value()
        .ok_or_else(|| vortex_err!("expected non-null variant"))?;
    let field = object
        .as_struct()
        .field("a")
        .ok_or_else(|| vortex_err!("expected field a"))?
        .as_variant()
        .value()
        .ok_or_else(|| vortex_err!("expected non-null field a"))?
        .cast(&i64_dtype())?;
    assert_eq!(field.as_primitive().typed_value::<i64>(), Some(1));

    let row1 = result.execute_scalar(1, &mut ctx)?;
    let value = row1
        .as_variant()
        .value()
        .ok_or_else(|| vortex_err!("expected non-null variant"))?
        .cast(&i64_dtype())?;
    assert_eq!(value.as_primitive().typed_value::<i64>(), Some(2));
    Ok(())
}

#[test]
fn converts_json_extension_input() -> VortexResult<()> {
    let storage = VarBinViewArray::from_iter_str(["1", "2"]).into_array();
    let input = json_input(storage)?;

    let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::NonNullable));
    assert_variant_i64_rows(&result, &[Some(1), Some(2)])
}

#[test]
fn dict_encoded_input_converts_each_row() -> VortexResult<()> {
    // A dictionary-encoded JSON column exercises the dict-pushdown / canonicalization path:
    // `json_to_variant` is not null-sensitive, so it pushes into the dict values (canonical
    // JSON extension values) where the kernel fires; either way every row must convert correctly.
    let values = json_input(VarBinViewArray::from_iter_str(["1", "2"]).into_array())?;
    let codes = PrimitiveArray::from_iter([0u8, 1, 0, 1, 0]).into_array();
    let input = DictArray::try_new(codes, values)?.into_array();

    let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::NonNullable));
    assert_variant_i64_rows(&result, &[Some(1), Some(2), Some(1), Some(2), Some(1)])
}

#[test]
fn null_rows_stay_null_and_json_null_becomes_variant_null() -> VortexResult<()> {
    let input = json_input(
        VarBinViewArray::from_iter_nullable_str([Some("1"), None, Some("null")]).into_array(),
    )?;

    let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    let mut ctx = SESSION.create_execution_ctx();
    assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
    assert_nth_scalar_is_null!(result, 1, &mut ctx);
    let row2 = result.execute_scalar(2, &mut ctx)?;
    assert!(!row2.is_null(), "JSON null must not be a row null");
    assert_eq!(row2.as_variant().is_variant_null(), Some(true));
    Ok(())
}

#[test]
fn invalid_json_errors() -> VortexResult<()> {
    let input =
        json_input(VarBinViewArray::from_iter_str([r#"{"a": 1}"#, r#"{"a":"#]).into_array())?;

    let err = execute_json_to_variant(input, ShreddingSpec::empty()).unwrap_err();
    assert!(!err.to_string().is_empty());
    Ok(())
}

#[test]
fn shredding_produces_typed_value_child() -> VortexResult<()> {
    let input = json_input(
        VarBinViewArray::from_iter_str([
            r#"{"a": 1, "b": "x"}"#,
            r#"{"a": 2, "b": "y"}"#,
            r#"{"a": "not-a-number", "b": "z"}"#,
            r#"{"b": "missing-a"}"#,
        ])
        .into_array(),
    )?;

    let result = execute_json_to_variant(input, shred_field_as_i64("a")?)?;

    assert!(
        result.as_::<ParquetVariant>().typed_value_array().is_some(),
        "expected shredded typed_value child"
    );

    // The canonical form must expose field `a` through the shredded tree.
    let mut ctx = SESSION.create_execution_ctx();
    let Canonical::Variant(canonical) = result.clone().execute::<Canonical>(&mut ctx)? else {
        vortex_bail!("expected canonical variant array");
    };
    let shredded = canonical
        .shredded()
        .ok_or_else(|| vortex_err!("expected canonical shredded child"))?
        .clone()
        .execute::<StructArray>(&mut ctx)?;
    assert!(shredded.unmasked_field_by_name_opt("a").is_some());

    // Typed extraction must serve shredded rows and fall back for mismatched rows.
    let typed = result
        .clone()
        .apply(&variant_get(
            root(),
            VariantPath::field("a"),
            Some(i64_dtype()),
        ))?
        .execute::<ArrayRef>(&mut ctx)?;
    assert_arrays_eq!(
        typed,
        PrimitiveArray::from_option_iter([Some(1i64), Some(2), None, None]),
        &mut ctx
    );

    // Mismatched rows keep their original value through the variant fallback.
    let untyped = result
        .apply(&variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Utf8(Nullability::Nullable)),
        ))?
        .execute::<ArrayRef>(&mut ctx)?;
    let row2 = untyped.execute_scalar(2, &mut ctx)?;
    assert_eq!(
        row2.as_utf8().value().map(|value| value.to_string()),
        Some("not-a-number".to_string())
    );
    Ok(())
}

#[test]
fn shredding_preserves_null_rows() -> VortexResult<()> {
    let input = json_input(
        VarBinViewArray::from_iter_nullable_str([Some(r#"{"a": 1}"#), None, Some(r#"{"a": 3}"#)])
            .into_array(),
    )?;

    let result = execute_json_to_variant(input, shred_field_as_i64("a")?)?;

    assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
    let mut ctx = SESSION.create_execution_ctx();
    assert_nth_scalar_is_null!(result, 1, &mut ctx);
    let typed = result
        .apply(&variant_get(
            root(),
            VariantPath::field("a"),
            Some(i64_dtype()),
        ))?
        .execute::<ArrayRef>(&mut ctx)?;
    assert_arrays_eq!(
        typed,
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn shredding_root_path_shreds_top_level_values() -> VortexResult<()> {
    let input =
        json_input(VarBinViewArray::from_iter_str(["1", "2", r#""not-a-number""#]).into_array())?;
    let spec = ShreddingSpec::try_new([(VariantPath::root(), i64_dtype())])?;

    let result = execute_json_to_variant(input, spec)?;

    assert!(
        result.as_::<ParquetVariant>().typed_value_array().is_some(),
        "expected shredded typed_value child"
    );
    assert_variant_i64_rows(&result.slice(0..2)?, &[Some(1), Some(2)])?;
    let mut ctx = SESSION.create_execution_ctx();
    let row2 = result.execute_scalar(2, &mut ctx)?;
    let value = row2
        .as_variant()
        .value()
        .ok_or_else(|| vortex_err!("expected non-null variant"))?;
    assert_eq!(
        value.as_utf8().value().map(|value| value.to_string()),
        Some("not-a-number".to_string())
    );
    Ok(())
}
