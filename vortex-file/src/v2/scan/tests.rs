// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use futures::TryStreamExt;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowScanExt;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::VTable as _;
use vortex_layout::layouts::list::List;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::ScanDriver;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

async fn file() -> VortexResult<VortexFile> {
    let chunks = 8;
    let numbers = ChunkedArray::from_iter(
        (0..chunks).map(|chunk| buffer![chunk, chunk + 1, chunk + 2, chunk + 3].into_array()),
    )
    .into_array();
    let strings = ChunkedArray::from_iter(
        (0..chunks).map(|_| VarBinArray::from(vec!["a", "bb", "ccc", "dddd"]).into_array()),
    )
    .into_array();
    let array = StructArray::from_fields(&[("numbers", numbers), ("strings", strings)])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf)
}

fn driver() -> ScanDriver {
    ScanDriver::new(TokioRuntime::current(), 2, true)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_batches_match_arrays() -> VortexResult<()> {
    let file = file().await?;
    let scan = || ScanBuilder::new(SESSION.clone(), file.scan_file());
    let schema = Arc::new(SESSION.arrow().to_arrow_schema(scan().dtype())?);
    let field = Field::new_struct("", schema.fields().clone(), false);

    let arrays: Vec<ArrayRef> = driver().run(scan()).try_collect().await?;
    let mut ctx = SESSION.create_execution_ctx();
    let expected = arrays
        .into_iter()
        .map(|array| {
            let arrow = array.execute_arrow(Some(&field), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let derived: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), None)
        .try_collect()
        .await?;
    let given: Vec<RecordBatch> = driver()
        .run_record_batches(scan(), Some(schema))
        .try_collect()
        .await?;

    assert_eq!(derived, expected);
    assert_eq!(given, expected);
    Ok(())
}

/// A struct of `rows` rows: an integer, and lists of 0 to 4 elements: nullable lists of
/// integers, every seventh null, lists of a few distinct strings, and lists of structs.
fn lists(rows: std::ops::Range<i32>) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut offsets = vec![0i64];
    for row in rows.clone() {
        offsets.push(offsets[offsets.len() - 1] + i64::from(row % 5));
    }
    let total = offsets[offsets.len() - 1];
    let offsets = Buffer::from_iter(offsets).into_array();
    let strings = || {
        VarBinArray::from_iter(
            (0..total).map(|i| Some(["a", "bb", "ccc"][(i % 3) as usize])),
            DType::Utf8(false.into()),
        )
        .into_array()
    };
    let lists = ListArray::try_new(
        Buffer::from_iter(0..total).into_array(),
        offsets.clone(),
        Validity::from_iter(rows.clone().map(|row| row % 7 != 0)),
        &mut ctx,
    )?;
    let names = ListArray::try_new(strings(), offsets.clone(), Validity::NonNullable, &mut ctx)?;
    let points = StructArray::from_fields(&[
        (
            "x",
            Buffer::from_iter((0..total).map(|i| i * 2)).into_array(),
        ),
        ("label", strings()),
    ])?;
    let points = ListArray::try_new(
        points.into_array(),
        offsets,
        Validity::NonNullable,
        &mut ctx,
    )?;
    Ok(StructArray::from_fields(&[
        ("ints", Buffer::from_iter(rows).into_array()),
        ("lists", lists.into_array()),
        ("names", names.into_array()),
        ("points", points.into_array()),
    ])?
    .into_array())
}

/// A file whose list column is written as a list layout, which the V2 scan reads under filters
/// and row ranges.
#[rstest]
#[case::whole(None, None)]
#[case::row_range(Some(9_000..21_500), None)]
#[case::sparse_filter(None, Some(eq(get_item("ints", root()), lit(17_001i32))))]
#[case::range_filter(Some(1_000..29_000), Some(gt(get_item("ints", root()), lit(15_000i32))))]
#[case::list_filter(None, Some(is_null(get_item("lists", root()))))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_layouts_scan_to_the_rows_written(
    #[case] row_range: Option<std::ops::Range<u64>>,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    let input = ChunkedArray::from_iter([lists(0..12_000)?, lists(12_000..30_000)?]).into_array();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, input.clone().to_array_stream())
        .await?;
    let file = SESSION.open_options().open_buffer(buf)?;
    let has_list_layout = file
        .footer()
        .layout()
        .depth_first_traversal()
        .map(|layout| layout.map(|layout| layout.encoding_id()))
        .collect::<VortexResult<Vec<_>>>()?
        .contains(&List.id());
    assert!(
        has_list_layout,
        "the file writes its list column as a list layout"
    );

    let mut scan = ScanBuilder::new(SESSION.clone(), file.scan_file())
        .with_some_filter(filter.clone().map(|f| f.bind(input.dtype())).transpose()?);
    if let Some(row_range) = row_range.clone() {
        scan = scan.with_row_range(row_range);
    }
    let dtype = scan.dtype().clone();
    let arrays: Vec<ArrayRef> = driver().run(scan).try_collect().await?;
    let actual = ChunkedArray::try_new(arrays, dtype)?.into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let row_range = row_range.unwrap_or(0..input.len() as u64);
    let mut keep = Mask::from_iter((0..input.len() as u64).map(|row| row_range.contains(&row)));
    if let Some(filter) = &filter {
        let matches = input
            .clone()
            .apply(filter)?
            .fill_null(false)?
            .execute::<Mask>(&mut ctx)?;
        keep = &keep & &matches;
    }
    let expected = input.filter(keep)?;
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}
