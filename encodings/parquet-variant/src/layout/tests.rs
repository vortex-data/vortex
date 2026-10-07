// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::variant_get;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_json::Json;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use super::ParquetVariantLayoutEncoding;
use super::expr::storage_field_masks;
use crate::ShreddingSpec;
use crate::json_to_variant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    crate::initialize(&session);
    session
});

/// Documents covering the shredding cases: `a` is shredded as an integer but one document holds a
/// string, `b.c` is always a string where present, `b.d` mixes types, `e` is never shredded, and
/// some documents are not objects at all.
const DOCUMENTS: &[&str] = &[
    r#"{"a": 1, "b": {"c": "x", "d": true}, "e": "keep"}"#,
    r#"{"a": 2, "b": {"c": "y", "d": "mixed"}, "e": {"f": 7}}"#,
    r#"{"a": "three", "b": {"c": "z"}, "e": [1, 2]}"#,
    r#"[1, 2]"#,
    r#"{"b": 5}"#,
    r#"{"a": 6, "b": {"c": "y", "d": false}}"#,
    r#"null"#,
];

fn path(fields: &[&str]) -> VariantPath {
    VariantPath::new(fields.iter().map(|f| VariantPathElement::field(*f)))
}

fn nullable(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::Nullable)
}

fn shredding() -> VortexResult<ShreddingSpec> {
    ShreddingSpec::try_new([
        (path(&["a"]), nullable(PType::I64)),
        (path(&["b", "c"]), DType::Utf8(Nullability::Nullable)),
        (path(&["b", "d"]), DType::Bool(Nullability::Nullable)),
    ])
}

/// A `{data: variant}` table holding the documents, shredded as [`shredding`].
fn table() -> VortexResult<ArrayRef> {
    let json = ExtensionArray::try_new_from_vtable(
        Json,
        EmptyMetadata,
        VarBinViewArray::from_iter_str(DOCUMENTS.iter().copied()).into_array(),
    )?
    .into_array();
    let data = json
        .apply(&json_to_variant(root(), shredding()?))?
        .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;
    Ok(StructArray::from_fields(&[("data", data)])?.into_array())
}

async fn write_file(table: ArrayRef) -> VortexResult<VortexFile> {
    let mut bytes = ByteBufferMut::empty();
    SESSION
        .write_options()
        .disable_editions()
        .write(&mut bytes, table.to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(bytes)
}

#[tokio::test]
async fn writes_parquet_variant_layout() -> VortexResult<()> {
    let file = write_file(table()?).await?;
    let tree = file.footer().layout().display_tree().to_string();
    assert!(
        tree.contains("data: vortex.parquet_variant"),
        "unexpected layout tree:\n{tree}"
    );
    // Every shredded leaf is a column of its own.
    assert!(tree.contains("typed_value: vortex.struct"), "{tree}");

    // Only `b.c` never holds a value that does not fit its shredded type.
    let data = file
        .footer()
        .layout()
        .slot(1)?
        .ok_or_else(|| vortex_err!("missing data column"))?;
    let data = data.as_::<ParquetVariantLayoutEncoding>();
    assert_eq!(
        data.typed_paths(),
        [vec![FieldName::from("b"), FieldName::from("c")]]
    );
    Ok(())
}

#[tokio::test]
async fn roundtrips_variant_values() -> VortexResult<()> {
    let expected = table()?;
    let file = write_file(expected.clone()).await?;
    let actual = file.scan()?.into_array_stream()?.read_all().await?;
    assert_arrays_eq!(expected, actual, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Scans evaluate `variant_get` over the layout's storage columns. Whatever storage the rewrite
/// reads, the result must match evaluating the expression over the in-memory Variant values.
#[rstest]
// Fully typed shredded path, served from its typed column.
#[case(variant_get(col("data"), path(&["b", "c"]), Some(DType::Utf8(Nullability::Nullable))))]
// Shredded path whose residual holds a mismatched value.
#[case(variant_get(col("data"), path(&["a"]), Some(nullable(PType::I64))))]
#[case(variant_get(col("data"), path(&["b", "d"]), Some(DType::Bool(Nullability::Nullable))))]
// Shredded path requested as another type.
#[case(variant_get(col("data"), path(&["a"]), Some(DType::Utf8(Nullability::Nullable))))]
#[case(variant_get(col("data"), path(&["b", "c"]), Some(nullable(PType::I32))))]
// Paths the shredded tree does not reach.
#[case(variant_get(col("data"), path(&["e"]), Some(DType::Utf8(Nullability::Nullable))))]
#[case(variant_get(col("data"), path(&["e", "f"]), Some(nullable(PType::I64))))]
#[case(variant_get(
    col("data"),
    VariantPath::new([VariantPathElement::field("e"), VariantPathElement::index(1)]),
    Some(nullable(PType::I64)),
))]
#[case(variant_get(col("data"), path(&["b", "c", "x"]), Some(DType::Utf8(Nullability::Nullable))))]
#[case(variant_get(col("data"), path(&["missing"]), Some(nullable(PType::I64))))]
#[case(variant_get(
    col("data"),
    VariantPath::new([VariantPathElement::index(0)]),
    Some(nullable(PType::I64)),
))]
// Variant results.
#[case(variant_get(
    variant_get(col("data"), path(&["b"]), None),
    path(&["c"]),
    Some(DType::Utf8(Nullability::Nullable)),
))]
#[case(variant_get(
    variant_get(col("data"), path(&["e"]), None),
    path(&["f"]),
    Some(nullable(PType::I64)),
))]
#[tokio::test]
async fn projects_variant_paths(#[case] expr: Expression) -> VortexResult<()> {
    let table = table()?;
    let mut ctx = SESSION.create_execution_ctx();
    let expected = table.clone().apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

    let file = write_file(table).await?;
    let actual = file
        .scan()?
        .with_projection(expr.bind(file.dtype())?)
        .into_array_stream()?
        .read_all()
        .await?;
    assert_arrays_eq!(expected, actual, &mut ctx);
    Ok(())
}

#[rstest]
#[case(eq(
    variant_get(col("data"), path(&["b", "c"]), Some(DType::Utf8(Nullability::Nullable))),
    lit("y"),
), 2)]
#[case(eq(
    variant_get(col("data"), path(&["a"]), Some(nullable(PType::I64))),
    lit(2i64),
), 1)]
#[case(eq(
    variant_get(col("data"), path(&["e", "f"]), Some(nullable(PType::I64))),
    lit(7i64),
), 1)]
#[tokio::test]
async fn filters_variant_paths(
    #[case] filter: Expression,
    #[case] rows: usize,
) -> VortexResult<()> {
    let file = write_file(table()?).await?;
    let actual = file
        .scan()?
        .with_filter(filter.bind(file.dtype())?)
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(actual.len(), rows);
    Ok(())
}

fn storage_dtype() -> DType {
    let utf8 = DType::Utf8(Nullability::Nullable);
    let binary = DType::Binary(Nullability::Nullable);
    let wrapper = |typed_value: DType| {
        DType::Struct(
            [("value", binary.clone()), ("typed_value", typed_value)]
                .into_iter()
                .collect(),
            Nullability::NonNullable,
        )
    };
    let b = DType::Struct(
        [("c", wrapper(utf8))].into_iter().collect(),
        Nullability::Nullable,
    );
    DType::Struct(
        [
            ("metadata", DType::Binary(Nullability::NonNullable)),
            ("value", binary.clone()),
            (
                "typed_value",
                DType::Struct(
                    [("b", wrapper(b))].into_iter().collect(),
                    Nullability::Nullable,
                ),
            ),
        ]
        .into_iter()
        .collect(),
        Nullability::NonNullable,
    )
}

fn prefix(fields: &[&str]) -> FieldMask {
    FieldMask::Prefix(FieldPath::from_iter(fields.iter().map(|f| Field::from(*f))))
}

#[rstest]
#[case::fully_typed(&["b", "c"], vec![prefix(&["typed_value", "b", "typed_value", "c", "typed_value"])])]
#[case::object(&["b"], vec![prefix(&["metadata"]), prefix(&["typed_value", "b"])])]
#[case::unshredded_child(
    &["b", "x"],
    vec![prefix(&["metadata"]), prefix(&["typed_value", "b", "value"])],
)]
#[case::unshredded_root_field(&["x"], vec![prefix(&["metadata"]), prefix(&["value"])])]
fn maps_variant_paths_to_storage_columns(
    #[case] variant_path: &[&str],
    #[case] expected: Vec<FieldMask>,
) {
    let typed_paths = vec![vec![FieldName::from("b"), FieldName::from("c")]];
    let masks = storage_field_masks(&[prefix(variant_path)], &storage_dtype(), &typed_paths);
    assert_eq!(masks, expected);
}

#[test]
fn whole_variant_maps_to_all_storage_columns() {
    let masks = storage_field_masks(&[FieldMask::All], &storage_dtype(), &[]);
    assert_eq!(masks, vec![FieldMask::All]);
}
