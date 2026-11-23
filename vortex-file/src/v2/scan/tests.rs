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
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_arrow::ArrowScanExt;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::metrics::ScanUsage;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_scan_records_usage() -> VortexResult<()> {
    let file = file().await?;
    let usage = Arc::new(ScanUsage::default());
    let tracked = file.clone().with_usage(Arc::clone(&usage));

    // Unpolled scan stream must not record file or split execution events.
    let unpolled = driver().run(ScanBuilder::new(SESSION.clone(), tracked.scan_file()));
    drop(unpolled);
    let before = usage.snapshot();
    assert_eq!(before.file_scans_attempted, 0);
    assert_eq!(before.splits_attempted, 0);

    // Normal V2 scan records file attempt, split attempts, and segment reads.
    let arrays: Vec<ArrayRef> = driver()
        .run(ScanBuilder::new(SESSION.clone(), tracked.scan_file()))
        .try_collect()
        .await?;
    assert_eq!(arrays.iter().map(|a| a.len()).sum::<usize>(), 32);
    let after_normal = usage.snapshot();
    assert_eq!(after_normal.file_scans_attempted, 1);
    assert_eq!(after_normal.file_scans_pruned, 0);
    assert_eq!(after_normal.splits_attempted, 1);
    assert!(after_normal.segments_requested > 0);
    assert!(after_normal.bytes_requested >= after_normal.segments_requested);

    // Whole-file pruning via zone maps records file_scans_pruned.
    let pruned: Vec<ArrayRef> = driver()
        .run(
            ScanBuilder::new(SESSION.clone(), tracked.scan_file())
                .with_some_filter(Some(gt(col("numbers"), lit(1000i32)).bind(file.dtype())?)),
        )
        .try_collect()
        .await?;
    assert!(pruned.is_empty());
    let after_prune = usage.snapshot().since(&after_normal);
    assert_eq!(after_prune.file_scans_attempted, 1);
    assert_eq!(after_prune.file_scans_pruned, 1);

    // Predicate that survives zone maps ("b" is within ["a", "dddd"]) but matches no rows
    // records empty_after_predicate without file-level pruning.
    let before_pred = usage.snapshot();
    let empty: Vec<ArrayRef> = driver()
        .run(
            ScanBuilder::new(SESSION.clone(), tracked.scan_file())
                .with_some_filter(Some(eq(col("strings"), lit("b")).bind(file.dtype())?)),
        )
        .try_collect()
        .await?;
    assert!(empty.is_empty());
    let after_pred = usage.snapshot().since(&before_pred);
    assert_eq!(after_pred.file_scans_attempted, 1);
    assert_eq!(after_pred.file_scans_pruned, 0);
    assert_eq!(after_pred.splits_attempted, 1);
    assert_eq!(after_pred.splits_empty_after_predicate, 1);

    // Statistics rejecting only a partial row range count as a pruned split, not whole-file pruning.
    let before_partial = usage.snapshot();
    let partial_empty: Vec<ArrayRef> = driver()
        .run(
            ScanBuilder::new(SESSION.clone(), tracked.scan_file())
                .with_row_range(0..1)
                .with_some_filter(Some(
                    gt(col("numbers"), lit(1_000_000i32)).bind(file.dtype())?,
                )),
        )
        .try_collect()
        .await?;
    assert!(partial_empty.is_empty());
    let after_partial = usage.snapshot().since(&before_partial);
    assert_eq!(after_partial.file_scans_attempted, 1);
    assert_eq!(after_partial.file_scans_pruned, 0);
    assert_eq!(after_partial.splits_attempted, 1);
    assert_eq!(after_partial.splits_pruned, 1);

    // Repeated execution increments file_scans_attempted again.
    let before_repeat = usage.snapshot();
    let _again: Vec<ArrayRef> = driver()
        .run(ScanBuilder::new(SESSION.clone(), tracked.scan_file()))
        .try_collect()
        .await?;
    let after_repeat = usage.snapshot().since(&before_repeat);
    assert_eq!(after_repeat.file_scans_attempted, 1);
    assert_eq!(after_repeat.splits_attempted, 1);
    Ok(())
}
