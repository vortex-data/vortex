// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stats::Precision;
use vortex_arrow::ArrowScanExt;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_scan::PartitionRef;
use vortex_scan::ScanRequest;
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

/// A file's plan data source scans the rows the request asks for, as a V2 scan built from
/// the file directly does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_data_source_scans_like_the_file() -> VortexResult<()> {
    let file = file().await?;
    let source = file.plan_data_source();
    assert_eq!(source.row_count(), Precision::exact(file.row_count()));

    let request = || ScanRequest {
        projection: get_item("numbers", root()),
        filter: Some(gt(get_item("numbers", root()), lit(4))),
        row_range: Some(4..28),
        ..Default::default()
    };
    let dtype = file.dtype().clone();
    let expected = ScanBuilder::new(SESSION.clone(), file.scan_file())
        .with_projection(request().projection.bind(&dtype)?)
        .with_some_filter(request().filter.map(|f| f.bind(&dtype)).transpose()?)
        .with_row_range(4..28);
    let scan_dtype = expected.dtype().clone();
    let collect = |arrays: Vec<ArrayRef>| ChunkedArray::try_new(arrays, scan_dtype.clone());
    let expected = collect(driver().run(expected).try_collect().await?)?;

    let partitions: Vec<PartitionRef> = source
        .scan(request())
        .await?
        .partitions()
        .try_collect()
        .await?;
    assert_eq!(partitions.len(), 1);
    let executed = collect(
        partitions
            .into_iter()
            .next()
            .vortex_expect("one partition")
            .execute()?
            .try_collect()
            .await?,
    )?;
    // The source's scan is unordered, so compare the rows in sorted order.
    let mut ctx = SESSION.create_execution_ctx();
    let mut sorted = |array: ChunkedArray| -> VortexResult<Vec<i32>> {
        let mut values = array
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?
            .as_slice::<i32>()
            .to_vec();
        values.sort_unstable();
        Ok(values)
    };
    assert_eq!(sorted(executed)?, sorted(expected)?);

    Ok(())
}
