// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::thread;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::session::LayoutSession;
use vortex_scan::planning::drive::PlanDriver;
use vortex_scan::planning::drive::ScopedBatch;
use vortex_scan::planning::drive::SequentialDriver;
use vortex_scan::planning::gather::Then;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use super::ScanDriver;
use super::collect;
use super::gather;
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

/// Rows of [`counting_file`]: enough for the writer to split them into many chunks.
const COUNTED: u64 = 1 << 20;

/// A file of `n`, counting from zero, beside `noise` that does not compress, so each chunk
/// needs reads of its own.
async fn counting_file() -> VortexResult<ByteBuffer> {
    let noise = (0..COUNTED).map(|n| n.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17));
    let array = StructArray::from_fields(&[
        ("n", PrimitiveArray::from_iter(0..COUNTED).into_array()),
        ("noise", PrimitiveArray::from_iter(noise).into_array()),
    ])?;
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.into_array().to_array_stream())
        .await?;
    Ok(buf.freeze())
}

/// Serves `bytes`, answering reads in the first half of the file last, so the chunks there
/// finish after the chunks behind them.
struct SlowStart(ByteBuffer);

impl VortexReadAt for SlowStart {
    fn concurrency(&self) -> usize {
        16
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.0.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let read = self.0.read_at(offset, length, alignment);
        let delay = if offset < self.0.len() as u64 / 2 {
            Duration::from_millis(20)
        } else {
            Duration::ZERO
        };
        async move {
            tokio::time::sleep(delay).await;
            read.await
        }
        .boxed()
    }
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

/// Per-batch work, such as executing to Arrow, must run on the driver's tasks: on the thread
/// polling the stream it is serialised, which is what slowed the decompress benchmark down.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batches_are_mapped_off_the_polling_thread() -> VortexResult<()> {
    let file = file().await?;
    let polling = thread::current().id();
    let mapped_on = Arc::new(Mutex::new(Vec::new()));
    let map_fn = {
        let mapped_on = Arc::clone(&mapped_on);
        Arc::new(move |batch: ArrayRef| {
            mapped_on.lock().push(thread::current().id());
            Ok(batch)
        })
    };

    let batches: Vec<ArrayRef> = driver()
        .drive(ScanBuilder::new(SESSION.clone(), file.scan_file()), map_fn)
        .try_collect()
        .await?;

    let mapped_on = mapped_on.lock();
    assert!(!batches.is_empty());
    assert_eq!(mapped_on.len(), batches.len());
    assert!(mapped_on.iter().all(|thread| *thread != polling));
    Ok(())
}

/// Later chunks overtake earlier ones whose reads are slower, and `collect` still returns the
/// rows in order: index builders number rows by their position in its output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn collect_returns_rows_in_order_when_reads_finish_out_of_order() -> VortexResult<()> {
    let file = SESSION
        .open_options()
        .open(Arc::new(SlowStart(counting_file().await?)))
        .await?;
    let batches = collect(ScanBuilder::new(SESSION.clone(), file.scan_file())).await?;
    assert!(batches.len() > 1);
    let mut ctx = SESSION.create_execution_ctx();
    let mut found = Vec::new();
    for batch in batches {
        let n = batch
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name("n")?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        found.extend_from_slice(n.as_slice::<u64>());
    }
    assert_eq!(found, (0..COUNTED).collect::<Vec<_>>());
    Ok(())
}

/// A gathered scan does its reads as the plan's own fetches, through the driver, and hands its
/// rows, in order, to the continuation rather than to the plan's output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gathered_scan_hands_its_rows_to_the_continuation() -> VortexResult<()> {
    let file = SESSION
        .open_options()
        .open(Arc::new(SlowStart(counting_file().await?)))
        .await?;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let then: Then = {
        let seen = Arc::clone(&seen);
        Box::new(move |batches| {
            let mut ctx = SESSION.create_execution_ctx();
            for batch in batches {
                let n = batch
                    .execute::<StructArray>(&mut ctx)?
                    .unmasked_field_by_name("n")?
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?;
                seen.lock().extend_from_slice(n.as_slice::<u64>());
            }
            Ok(None)
        })
    };
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..0,
    };
    let root = gather(
        scope,
        ScanBuilder::new(SESSION.clone(), file.scan_file()),
        then,
    )?;
    let output: Vec<ScopedBatch> = SequentialDriver::new().execute(root).try_collect().await?;
    assert!(output.is_empty());
    assert_eq!(*seen.lock(), (0..COUNTED).collect::<Vec<_>>());
    Ok(())
}
