// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writing null-dominated columns with the sparse layout and reading them with the v2 scan.

use std::ops::Range;
use std::sync::LazyLock;

use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::sparse::Sparse;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::ScanDriver;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
use crate::WriteStrategyBuilder;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

/// Three full blocks and part of a fourth, so scans cross block boundaries.
const ROWS: usize = 3 * 8192 + 100;

/// `g` is present on every 50th row and holds `a`, null on every 150th row, and `b`. `dense` is
/// null on every 10th row, too dense for a sparse layout, and `id` is never null.
fn table() -> VortexResult<ArrayRef> {
    let rows = 0..ROWS;
    let present = |row: usize| row.is_multiple_of(50);
    let a = PrimitiveArray::new(
        rows.clone().map(|row| row as i64).collect::<vortex_buffer::Buffer<i64>>(),
        Validity::from_iter(rows.clone().map(|row| !row.is_multiple_of(150))),
    );
    let b = PrimitiveArray::new(
        rows.clone().map(|row| 2 * row as i64).collect::<vortex_buffer::Buffer<i64>>(),
        Validity::NonNullable,
    );
    let g = StructArray::try_new(
        ["a", "b"].into(),
        vec![a.into_array(), b.into_array()],
        ROWS,
        Validity::from_iter(rows.clone().map(present)),
    )?;
    let dense = PrimitiveArray::new(
        rows.clone().map(|row| row as i64).collect::<vortex_buffer::Buffer<i64>>(),
        Validity::from_iter(rows.clone().map(|row| !row.is_multiple_of(10))),
    );
    let id = PrimitiveArray::new(
        rows.map(|row| row as u64).collect::<vortex_buffer::Buffer<u64>>(),
        Validity::NonNullable,
    );
    Ok(StructArray::from_fields(&[
        ("g", g.into_array()),
        ("dense", dense.into_array()),
        ("id", id.into_array()),
    ])?
    .into_array())
}

async fn file(array: &ArrayRef) -> VortexResult<VortexFile> {
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(
            WriteStrategyBuilder::from_session(&SESSION)
                .with_sparse_layout(0.9)
                .build(),
        )
        .write(&mut buf, array.to_array_stream())
        .await?;
    SESSION.open_options().open_buffer(buf)
}

async fn scan(
    file: &VortexFile,
    projection: Option<Expression>,
    filter: Option<Expression>,
    rows: Option<Range<u64>>,
) -> VortexResult<ArrayRef> {
    let mut scan = ScanBuilder::new(SESSION.clone(), file.scan_file());
    let dtype = scan.dtype().clone();
    let mut output_dtype = dtype.clone();
    if let Some(projection) = projection {
        let projection = projection.bind(&dtype)?;
        output_dtype = projection.dtype().clone();
        scan = scan.with_projection(projection);
    }
    scan = scan.with_some_filter(filter.map(|filter| filter.bind(&dtype)).transpose()?);
    if let Some(rows) = rows {
        scan = scan.with_row_range(rows);
    }
    let batches: Vec<ArrayRef> = ScanDriver::new(TokioRuntime::current(), 2, true)
        .run(scan)
        .try_collect()
        .await?;
    Ok(ChunkedArray::try_new(batches, output_dtype)?.into_array())
}

/// The dtypes of every sparse layout in the file.
fn sparse_layout_dtypes(file: &VortexFile) -> VortexResult<Vec<DType>> {
    let mut dtypes = Vec::new();
    let mut stack = vec![file.footer().layout().to_layout()];
    while let Some(layout) = stack.pop() {
        if layout.is::<Sparse>() {
            dtypes.push(layout.dtype().clone());
        }
        stack.extend(layout.children()?);
    }
    Ok(dtypes)
}

/// Only `g` is null-dominated: neither its compacted fields nor `dense` become sparse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writes_only_null_dominated_columns_as_sparse() -> VortexResult<()> {
    let array = table()?;
    let file = file(&array).await?;
    let g_dtype = array
        .dtype()
        .as_struct_fields_opt()
        .and_then(|fields| fields.field("g"))
        .ok_or_else(|| vortex_error::vortex_err!("table has no g"))?;
    assert_eq!(sparse_layout_dtypes(&file)?, vec![g_dtype]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scans_whole_table() -> VortexResult<()> {
    let array = table()?;
    let file = file(&array).await?;
    let actual = scan(&file, None, None, None).await?;
    assert_arrays_eq!(actual, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A row range starting and ending inside blocks reads only the present rows inside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scans_row_range_across_blocks() -> VortexResult<()> {
    let array = table()?;
    let file = file(&array).await?;
    let rows = 5_003..17_011;
    let actual = scan(&file, None, None, Some(rows.clone())).await?;
    let expected = array.slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A nested field projects through the sparse layout onto its compacted values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn projects_nested_field() -> VortexResult<()> {
    let array = table()?;
    let file = file(&array).await?;
    let projection = get_item("a", get_item("g", root()));
    let actual = scan(&file, Some(projection.clone()), None, None).await?;
    let expected = array.apply_bound(&projection.bind(array.dtype())?)?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// A filter on a nested field keeps the present rows it holds for and none of the absent ones,
/// whose field is null.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filters_on_nested_field() -> VortexResult<()> {
    let array = table()?;
    let file = file(&array).await?;
    let threshold = 20_000;
    let filter = gt(get_item("a", get_item("g", root())), lit(threshold as i64));
    let actual = scan(&file, None, Some(filter), None).await?;
    let selected = Mask::from_indices(
        ROWS,
        (0..ROWS).filter(|&row| {
            row.is_multiple_of(50) && !row.is_multiple_of(150) && row > threshold
        }),
    );
    let expected = array.filter(selected)?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}
