// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads of shredded Variant columns stored in the Variant layout.
//!
//! The layout reader prunes the shredded tree to the paths an expression reads. Every read must
//! return what evaluating the same expression over the in-memory array returns.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use arrow_schema::DataType;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::variant_get;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_array::stream::ArrayStreamExt;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_session::VortexSession;

use crate::ParquetVariant;
use crate::vtable::tests::parquet_variant_file_session;

/// Rows exercising typed leaves, partially shredded objects, type mismatches, missing paths and
/// non-object values.
const ROWS: [&str; 8] = [
    r#"{"kind": "commit", "commit": {"collection": "post", "operation": "create", "rev": 1}, "time": 10}"#,
    r#"{"kind": "identity", "time": 11}"#,
    r#"{"kind": 5, "commit": "not an object", "time": "late"}"#,
    r#"{"commit": {"collection": 7, "operation": "delete"}, "time": 12}"#,
    r#"null"#,
    r#""a string""#,
    r#"{"kind": "commit", "commit": {"collection": "like", "record": {"a": "b"}}, "time": 13}"#,
    r#"{"kind": "commit", "commit": {"operation": "create"}}"#,
];

fn session() -> VortexResult<VortexSession> {
    let session = parquet_variant_file_session()?;
    crate::initialize(&session);
    Ok(session)
}

/// A `{data: Variant}` table of `copies` copies of [`ROWS`], shredding the commonly read paths.
fn table(session: &VortexSession, copies: usize) -> VortexResult<ArrayRef> {
    let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values(
        ROWS.iter().cycle().take(ROWS.len() * copies),
    ));
    let schema = ShreddedSchemaBuilder::new()
        .with_path("kind", &DataType::Utf8)?
        .with_path("commit.collection", &DataType::Utf8)?
        .with_path("commit.operation", &DataType::Utf8)?
        .with_path("time", &DataType::Int64)?
        .build();
    let variant = shred_variant(&json_to_variant(&json)?, &schema)?;
    let data = ParquetVariant::from_arrow_variant(&variant, &session.arrow())?;
    Ok(StructArray::try_from_iter([("data", data)])?.into_array())
}

async fn write(session: &VortexSession, table: ArrayRef) -> VortexResult<ByteBufferMut> {
    let mut bytes = ByteBufferMut::empty();
    let strategy = vortex_file::WriteStrategyBuilder::from_session(session).build();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, table.to_array_stream())
        .await?;
    Ok(bytes)
}

fn get(path: &str, dtype: DType) -> Expression {
    variant_get(
        get_item("data", root()),
        VariantPath::new(path.split('.').map(VariantPathElement::field)),
        Some(dtype),
    )
}

fn utf8(path: &str) -> Expression {
    get(path, DType::Utf8(Nullability::Nullable))
}

fn i64(path: &str) -> Expression {
    get(path, DType::Primitive(PType::I64, Nullability::Nullable))
}

#[rstest]
#[case::typed_leaf(utf8("kind"))]
#[case::through_partially_shredded_object(utf8("commit.collection"))]
#[case::unshredded_path(i64("commit.rev"))]
#[case::nested_unshredded_object(utf8("commit.record.a"))]
#[case::type_mismatch(utf8("time"))]
#[case::missing_shredded_field(utf8("missing"))]
#[case::several_paths(pack(
    [
        ("kind", utf8("kind")),
        ("collection", utf8("commit.collection")),
        ("operation", utf8("commit.operation")),
        ("time", i64("time")),
    ],
    Nullability::NonNullable,
))]
#[case::comparison(eq(utf8("kind"), lit("commit")))]
#[tokio::test]
async fn projection_matches_in_memory(
    #[case] expr: Expression,
    #[values(1, 2000)] copies: usize,
) -> VortexResult<()> {
    let session = session()?;
    let table = table(&session, copies)?;
    let bytes = write(&session, table.clone()).await?;
    let file = session.open_options().open_buffer(bytes)?;
    assert!(
        format!("{}", file.footer().layout().display_tree()).contains("vortex.variant"),
        "the file must store the column with the Variant layout"
    );

    let actual = file
        .scan()?
        .with_projection(expr.bind(file.dtype())?)
        .into_array_stream()?
        .read_all()
        .await?;

    let mut ctx = session.create_execution_ctx();
    let expected = table
        .apply(&expr)?
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn filter_matches_in_memory(#[values(1, 2000)] copies: usize) -> VortexResult<()> {
    let session = session()?;
    let table = table(&session, copies)?;
    let bytes = write(&session, table.clone()).await?;
    let file = session.open_options().open_buffer(bytes)?;

    let filter = and(
        eq(utf8("kind"), lit("commit")),
        eq(utf8("commit.operation"), lit("create")),
    );
    let projection = pack(
        [
            ("collection", utf8("commit.collection")),
            ("time", i64("time")),
        ],
        Nullability::NonNullable,
    );
    let actual = file
        .scan()?
        .with_filter(filter.bind(file.dtype())?)
        .with_projection(projection.bind(file.dtype())?)
        .into_array_stream()?
        .read_all()
        .await?;

    let mut ctx = session.create_execution_ctx();
    let mask = table
        .clone()
        .apply(&filter)?
        .execute::<Canonical>(&mut ctx)?
        .into_array()
        .null_as_false()
        .execute(&mut ctx)?;
    let expected = table
        .filter(mask)?
        .apply(&projection)?
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    assert_eq!(actual.len(), 2 * copies);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}
