// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use parquet_variant::Variant as PqVariant;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::unshred_variant;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::root;
use vortex_array::expr::variant_get;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_shred::VariantShred;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::variant::VariantShreddingPolicy;
use vortex_session::VortexSession;

use super::InferredShredding;
use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;
use crate::arrow::prepare_variant_for_export;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

const ROWS: [&str; 4] = [
    r#"{"kind":"commit","time_us":1,"commit":{"collection":"post","rev":"a"},"tags":["x"]}"#,
    r#"{"kind":"commit","time_us":2,"commit":{"collection":"like","rev":7}}"#,
    r#"{"kind":"identity","time_us":3,"mixed":"text"}"#,
    r#"{"kind":"account","time_us":"late","mixed":4}"#,
];

/// Values of every type in each field, so that each typed column takes some and not others.
const VALUES: [Option<&str>; 9] = [
    Some(r#"{"n":1,"s":"a","b":true,"d":0.5,"l":[1,"x",null],"o":{"n":1,"x":2}}"#),
    Some(r#"{"n":1.5,"s":7,"b":1,"d":2,"l":"x","o":{"x":3}}"#),
    Some(r#"{"n":true,"s":null,"b":"true","d":"0.5","l":[],"o":[1]}"#),
    Some(r#"{"n":300,"s":"a long string that is not inlined","b":false,"o":{"n":"1"}}"#),
    Some(r#"{"n":-9007199254740993,"l":[[1]],"o":{"n":2}}"#),
    Some(r#"{"n":null,"x":{"n":1}}"#),
    Some(r#"[{"n":1}]"#),
    Some("1"),
    None,
];

fn from_json<'a>(rows: impl IntoIterator<Item = Option<&'a str>>) -> VortexResult<ArrayRef> {
    let json: ArrowArrayRef = Arc::new(StringArray::from_iter(rows));
    ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())
}

fn variants() -> VortexResult<ArrayRef> {
    from_json(ROWS.map(Some))
}

fn utf8() -> DType {
    DType::Utf8(Nullability::Nullable)
}

fn int(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::Nullable)
}

fn boolean() -> DType {
    DType::Bool(Nullability::Nullable)
}

fn list(element: DType) -> DType {
    DType::List(Arc::new(element), Nullability::Nullable)
}

fn object(fields: &[(&str, DType)]) -> DType {
    DType::Struct(
        StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.clone()))),
        Nullability::Nullable,
    )
}

fn shred_into(array: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    VariantShred::try_new(array, dtype)?
        .into_array()
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
}

fn inferred(policy: &InferredShredding) -> VortexResult<Option<DType>> {
    policy.schema(&variants()?, &mut SESSION.create_execution_ctx())
}

fn shredded() -> VortexResult<ArrayRef> {
    let dtype =
        inferred(&InferredShredding::default())?.ok_or_else(|| vortex_err!("nothing to shred"))?;
    shred_into(variants()?, dtype)
}

/// The Parquet Variant storage of `array`.
fn storage(array: ArrayRef) -> VortexResult<ArrayRef> {
    prepare_variant_for_export(array, None, &mut SESSION.create_execution_ctx())
}

/// The value of each row of `array`, unshredded by the Arrow Variant reader, with the integers
/// of all widths written alike.
fn values(array: ArrayRef) -> VortexResult<Vec<Option<String>>> {
    let mut ctx = SESSION.create_execution_ctx();
    let variants = unshred_variant(
        &storage(array)?
            .as_::<ParquetVariant>()
            .to_arrow(&ArrowExportOptions::default(), &mut ctx)?,
    )?;
    (0..variants.len())
        .map(|idx| {
            if !variants.is_valid(idx) {
                return Ok(None);
            }
            let mut out = String::new();
            render(&variants.try_value(idx)?, &mut out);
            Ok(Some(out))
        })
        .collect()
}

#[expect(
    clippy::use_debug,
    reason = "the Debug of a Variant names its type and value"
)]
fn render(variant: &PqVariant<'_, '_>, out: &mut String) {
    match variant {
        PqVariant::Object(object) => {
            out.push('{');
            for (name, field) in object.iter() {
                let _ = write!(out, "{name:?}:");
                render(&field, out);
                out.push(',');
            }
            out.push('}');
        }
        PqVariant::List(list) => {
            out.push('[');
            for element in list.iter() {
                render(&element, out);
                out.push(',');
            }
            out.push(']');
        }
        PqVariant::Int8(_) | PqVariant::Int16(_) | PqVariant::Int32(_) | PqVariant::Int64(_) => {
            let _ = write!(out, "int {:?}", variant.as_int64());
        }
        other => {
            let _ = write!(out, "{other:?}");
        }
    }
}

/// The `typed_value` column of the shredded field `name` of `array`.
fn typed_field(array: ArrayRef, name: &str) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let typed_value = storage(array)?
        .as_::<ParquetVariant>()
        .typed_value()
        .ok_or_else(|| vortex_err!("not shredded"))?
        .clone()
        .execute::<StructArray>(&mut ctx)?;
    let field = typed_value
        .unmasked_field_by_name(name)?
        .clone()
        .execute::<StructArray>(&mut ctx)?;
    Ok(field.unmasked_field_by_name("typed_value")?.clone())
}

/// The names of the fields of the shredded child of `array`.
fn shredded_names(array: ArrayRef) -> VortexResult<Vec<String>> {
    let shredded = array
        .execute::<VariantArray>(&mut SESSION.create_execution_ctx())?
        .shredded()
        .ok_or_else(|| vortex_err!("not shredded"))?
        .dtype()
        .clone();
    let DType::Struct(fields, _) = shredded else {
        return Err(vortex_err!("shredded as {shredded}"));
    };
    Ok(fields.names().iter().map(|name| name.to_string()).collect())
}

fn get(array: ArrayRef, path: &[&str], dtype: DType) -> VortexResult<ArrayRef> {
    let path = VariantPath::new(path.iter().map(|name| (*name).into()));
    array
        .apply(&variant_get(root(), path, Some(dtype)))?
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
}

#[test]
fn infers_the_fields_of_one_type() -> VortexResult<()> {
    // `time_us`, `mixed` and `commit.rev` hold two types, and `tags` holds a list.
    assert_eq!(
        inferred(&InferredShredding::default())?,
        Some(object(&[
            ("commit", object(&[("collection", utf8())])),
            ("kind", utf8()),
        ]))
    );
    Ok(())
}

#[test]
fn infers_each_type_of_value() -> VortexResult<()> {
    let array = from_json([
        Some(r#"{"s":"a","b":true,"i":1,"d":0.5,"n":null}"#),
        Some(r#"{"s":"b","b":false,"i":300,"d":1.5,"n":null}"#),
    ])?;
    let inferred =
        InferredShredding::default().schema(&array, &mut SESSION.create_execution_ctx())?;
    assert_eq!(
        inferred,
        Some(object(&[
            ("b", boolean()),
            ("d", int(PType::F64)),
            ("i", int(PType::I64)),
            ("s", utf8()),
        ]))
    );
    Ok(())
}

#[test]
fn infers_the_fields_in_the_most_rows() -> VortexResult<()> {
    let policy = InferredShredding::default().with_max_fields(1);
    assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
    Ok(())
}

#[test]
fn infers_fields_that_enough_rows_have() -> VortexResult<()> {
    // `commit.collection` is in two rows of four.
    let policy = InferredShredding::default().with_min_presence_percent(75);
    assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
    Ok(())
}

#[test]
fn infers_fields_no_deeper_than_the_depth() -> VortexResult<()> {
    let policy = InferredShredding::default().with_max_depth(1);
    assert_eq!(inferred(&policy)?, Some(object(&[("kind", utf8())])));
    Ok(())
}

#[test]
fn infers_nothing_from_values_without_common_fields() -> VortexResult<()> {
    let array = from_json([Some("1"), Some("[1,2]"), Some("\"a\"")])?;
    let inferred =
        InferredShredding::default().schema(&array, &mut SESSION.create_execution_ctx())?;
    assert!(inferred.is_none());
    Ok(())
}

#[test]
fn shreds_by_the_given_schema() -> VortexResult<()> {
    let schema = object(&[("kind", utf8())]);
    assert_eq!(shredded_names(shred_into(variants()?, schema)?)?, ["kind"]);
    Ok(())
}

#[test]
fn shreds_by_the_inferred_schema() -> VortexResult<()> {
    assert_eq!(shredded_names(shredded()?)?, ["commit", "kind"]);
    Ok(())
}

#[rstest]
#[case::variant(DType::Variant(Nullability::Nullable))]
#[case::unsigned(int(PType::U8))]
#[case::decimal(DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable))]
#[case::nested(object(&[("o", list(DType::Variant(Nullability::Nullable)))]))]
fn rejects_leaves_of_other_types(#[case] leaf: DType) -> VortexResult<()> {
    assert!(shred_into(variants()?, object(&[("time_us", leaf)])).is_err());
    Ok(())
}

#[test]
fn does_not_shred_a_shredded_variant() -> VortexResult<()> {
    assert!(shred_into(shredded()?, object(&[("kind", utf8())])).is_err());
    Ok(())
}

#[rstest]
#[case::string(object(&[("s", utf8())]))]
#[case::binary(object(&[("s", DType::Binary(Nullability::Nullable))]))]
#[case::boolean(object(&[("b", boolean())]))]
#[case::int8(object(&[("n", int(PType::I8))]))]
#[case::int16(object(&[("n", int(PType::I16))]))]
#[case::int32(object(&[("n", int(PType::I32))]))]
#[case::int64(object(&[("n", int(PType::I64))]))]
#[case::float(object(&[("d", int(PType::F32))]))]
#[case::double(object(&[("d", int(PType::F64))]))]
#[case::nested(object(&[("o", object(&[("n", int(PType::I64))]))]))]
#[case::every_field(object(&[("o", object(&[("n", int(PType::I64)), ("x", int(PType::I64))]))]))]
#[case::list(object(&[("l", list(int(PType::I64)))]))]
#[case::nested_list(object(&[("l", list(list(int(PType::I64))))]))]
#[case::list_of_objects(list(object(&[("n", int(PType::I64))])))]
#[case::root(int(PType::I64))]
#[case::all(object(&[
    ("b", boolean()),
    ("d", int(PType::F64)),
    ("l", list(utf8())),
    ("n", int(PType::I64)),
    ("o", object(&[("n", utf8())])),
    ("s", utf8()),
]))]
fn shredding_keeps_every_value(#[case] schema: DType) -> VortexResult<()> {
    let array = from_json(VALUES)?;
    assert_eq!(values(shred_into(array.clone(), schema)?)?, values(array)?);
    Ok(())
}

#[test]
fn shreds_the_values_of_the_column_type() -> VortexResult<()> {
    // 1.5, true and a Variant null are not integers, and stay in the residual value.
    let shredded = shred_into(from_json(VALUES)?, object(&[("n", int(PType::I64))]))?;
    assert_arrays_eq!(
        typed_field(shredded, "n")?,
        PrimitiveArray::from_option_iter([
            Some(1i64),
            None,
            None,
            Some(300),
            Some(-9007199254740993),
            None,
            None,
            None,
            None,
        ]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn shreds_the_integers_that_fit_the_column() -> VortexResult<()> {
    let shredded = shred_into(from_json(VALUES)?, object(&[("n", int(PType::I8))]))?;
    assert_arrays_eq!(
        typed_field(shredded, "n")?,
        PrimitiveArray::from_option_iter([
            Some(1i8),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        ]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn shreds_booleans_alone() -> VortexResult<()> {
    let shredded = shred_into(from_json(VALUES)?, object(&[("b", boolean())]))?;
    assert_arrays_eq!(
        typed_field(shredded, "b")?,
        BoolArray::from_iter([
            Some(true),
            None,
            None,
            Some(false),
            None,
            None,
            None,
            None,
            None
        ]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn shreds_objects_with_many_fields_and_names() -> VortexResult<()> {
    // More than 255 fields, whose ids are two bytes wide.
    let fields = (0..300)
        .map(|idx| format!(r#""f{idx:03}":{idx}"#))
        .collect::<Vec<_>>()
        .join(",");
    let row = format!("{{{fields}}}");
    let array = from_json([Some(row.as_str()), Some(r#"{"f150":"x"}"#)])?;

    let schema = object(&[("f000", int(PType::I64)), ("f299", int(PType::I64))]);
    assert_eq!(values(shred_into(array.clone(), schema)?)?, values(array)?);
    Ok(())
}

#[rstest]
#[case::shredded(&["kind"])]
#[case::nested(&["commit", "collection"])]
#[case::residual(&["mixed"])]
#[case::residual_nested(&["commit", "rev"])]
#[case::missing(&["nope"])]
fn shredding_keeps_the_strings(#[case] path: &[&str]) -> VortexResult<()> {
    let expected = get(variants()?, path, utf8())?
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
    assert_arrays_eq!(
        get(shredded()?, path, utf8())?,
        expected,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
#[case::residual(&["time_us"])]
#[case::residual_nested(&["commit", "rev"])]
fn shredding_keeps_the_integers(#[case] path: &[&str]) -> VortexResult<()> {
    let int64 = int(PType::I64);
    let expected = get(variants()?, path, int64.clone())?
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
    assert_arrays_eq!(
        get(shredded()?, path, int64)?,
        expected,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}
