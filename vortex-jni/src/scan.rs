// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings mirroring `vortex-ffi`'s scan/partition API.
//!
//! The scan lifecycle is:
//! 1. `Java_dev_vortex_jni_NativeScan_create` → returns a [`NativeScan`] pointer.
//! 2. `Java_dev_vortex_jni_NativeScan_nextPartition` → repeatedly pulls partitions out
//!    of the scan stream. Returns `0` when exhausted.
//! 3. `Java_dev_vortex_jni_NativePartition_scanArrow` → consumes a partition into an
//!    `FFI_ArrowArrayStream` that Java imports via Arrow's C Data Interface.

use std::io::Cursor;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::sys::jboolean;
use jni::sys::jint;
use jni::sys::jlong;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::scan::DataSourceScan;
use vortex::scan::PartitionRef;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::POOL;
use crate::RUNTIME;
use crate::arrow_compat::rebase_offsets;
use crate::data_source::NativeDataSource;
use crate::errors::try_or_throw;
use crate::session::session_ref;

/// Opaque scan handle. Holds a three-state machine: either the scan is pending (not yet
/// realized as a stream), actively streaming partitions, or finished.
///
/// The scan's `split_concurrency`, if set, also bounds how many of each partition's arrays are
/// converted to Arrow ahead of Java, so the conversion stays within the scan's bound.
#[allow(dead_code)]
pub(crate) enum NativeScan {
    Pending(Box<dyn DataSourceScan>, Option<NonZeroUsize>),
    Started(PartitionStream, Option<NonZeroUsize>),
    Finished,
}

/// Opaque partition handle with the same three-state machine shape, carrying its scan's
/// `split_concurrency`.
#[allow(dead_code)]
pub(crate) enum NativePartition {
    Pending(PartitionRef, Option<NonZeroUsize>),
    Started(SendableArrayStream),
    Finished,
}

#[allow(clippy::too_many_arguments)]
fn build_scan_request(
    projection_ptr: jlong,
    filter_ptr: jlong,
    row_range_begin: jlong,
    row_range_end: jlong,
    selection_idx: &[u64],
    selection_roaring_bitmap: &[u8],
    selection_include: u8,
    limit: jlong,
    ordered: jboolean,
    split_concurrency: jint,
) -> VortexResult<ScanRequest> {
    let projection = if projection_ptr == 0 {
        root()
    } else {
        unsafe { &*(projection_ptr as *const Expression) }.clone()
    };

    let filter = if filter_ptr == 0 {
        None
    } else {
        Some(unsafe { &*(filter_ptr as *const Expression) }.clone())
    };

    let selection = match selection_include {
        0 => Selection::All,
        1 => Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::copy_from(
            selection_idx,
        ))?),
        2 => Selection::ExcludeByIndex(StrictSortedBuffer::try_new(Buffer::copy_from(
            selection_idx,
        ))?),
        3 => Selection::IncludeRoaring(deserialize_roaring_selection(selection_roaring_bitmap)?),
        4 => Selection::ExcludeRoaring(deserialize_roaring_selection(selection_roaring_bitmap)?),
        other => vortex_bail!("unknown selection include code: {other}"),
    };

    let row_range = (row_range_begin > 0 || row_range_end > 0).then_some(Range {
        start: row_range_begin as u64,
        end: row_range_end as u64,
    });

    let limit = (limit > 0).then_some(limit as u64);
    let split_concurrency =
        NonZeroUsize::new(usize::try_from(split_concurrency).map_err(|_| {
            vortex_err!("split concurrency must not be negative, got {split_concurrency}")
        })?);

    Ok(ScanRequest {
        projection,
        filter,
        row_range,
        selection,
        ordered,
        limit,
        partition_selection: Selection::All,
        partition_range: None,
        split_concurrency,
    })
}

fn deserialize_roaring_selection(bytes: &[u8]) -> VortexResult<roaring::RoaringTreemap> {
    if bytes.is_empty() {
        vortex_bail!("serialized roaring row selection must not be empty");
    }
    let cursor = Cursor::new(bytes);
    Ok(roaring::RoaringTreemap::deserialize_from(cursor)?)
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeScan_create(
    mut env: EnvUnowned,
    _class: JClass,
    data_source_ptr: jlong,
    projection_ptr: jlong,
    filter_ptr: jlong,
    row_range_begin: jlong,
    row_range_end: jlong,
    selection_indices: JLongArray,
    selection_roaring_bitmap: JByteArray,
    selection_include: jni::sys::jbyte,
    limit: jlong,
    ordered: jboolean,
    split_concurrency: jint,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let ds = unsafe { NativeDataSource::from_ptr(data_source_ptr) };

        let selection_idx: Vec<u64> = if selection_indices.is_null() {
            Vec::new()
        } else {
            let elements = unsafe {
                selection_indices.get_elements(env, jni::objects::ReleaseMode::NoCopyBack)
            }?;
            let mut out: Vec<u64> = Vec::with_capacity(elements.len());
            for v in elements.iter() {
                if *v < 0 {
                    throw_runtime!("row selection index must be non-negative");
                }
                out.push(*v as u64);
            }
            out
        };

        let selection_roaring_bitmap: Vec<u8> = if selection_roaring_bitmap.is_null() {
            Vec::new()
        } else {
            env.convert_byte_array(&selection_roaring_bitmap)?
        };

        let request = build_scan_request(
            projection_ptr,
            filter_ptr,
            row_range_begin,
            row_range_end,
            &selection_idx,
            &selection_roaring_bitmap,
            selection_include as u8,
            limit,
            ordered,
            split_concurrency,
        )?;

        let split_concurrency = request.split_concurrency;
        let scan = RUNTIME.block_on(async { ds.inner().scan(request).await })?;
        Ok(Box::into_raw(Box::new(NativeScan::Pending(scan, split_concurrency))) as jlong)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeScan_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer == 0 {
        return;
    }
    drop(unsafe { Box::from_raw(pointer as *mut NativeScan) });
}

/// Write the scan's DType as an Arrow schema to the FFI struct at `schema_addr`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeScan_arrowSchema(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    pointer: jlong,
    schema_addr: jlong,
) {
    try_or_throw(&mut env, |_| {
        if schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let session = unsafe { session_ref(session_ptr) };
        let scan = unsafe { &*(pointer as *const NativeScan) };
        let NativeScan::Pending(scan, _) = scan else {
            throw_runtime!("schema unavailable: scan already started");
        };
        let arrow_schema = session.arrow().to_arrow_schema(scan.dtype())?;
        let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe {
            ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
        }
        Ok(())
    });
}

/// Get the estimated partition count. Writes `[rows, cardinality]` into `out`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeScan_partitionCount(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    out: JLongArray,
) {
    try_or_throw(&mut env, |env| {
        let scan = unsafe { &*(pointer as *const NativeScan) };
        let NativeScan::Pending(scan, _) = scan else {
            throw_runtime!("partition count unavailable: scan already started");
        };
        let (rows, cardinality) = match scan.partition_count() {
            Precision::Exact(v) => (v as jlong, 2),
            Precision::Inexact(v) => (v as jlong, 1),
            Precision::Absent => (0, 0),
        };
        out.set_region(env, 0, &[rows, cardinality])?;
        Ok(())
    });
}

/// Advance the scan to its next partition. Returns `0` when exhausted.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeScan_nextPartition(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    try_or_throw(&mut env, |_| unsafe {
        let slot = pointer as *mut NativeScan;
        let owned = ptr::read(slot);

        // `owned` has been `ptr::read`-copied out of `slot`; the bit pattern at `slot`
        // must be overwritten with a valid value on every path, including errors, or
        // `NativeScan::free` will double-drop. Park `slot` as `Finished` upfront and
        // only rewrite it on the happy path where we want to keep streaming.
        ptr::write(slot, NativeScan::Finished);

        let (mut stream, split_concurrency) = match owned {
            NativeScan::Pending(scan, split_concurrency) => (scan.partitions(), split_concurrency),
            NativeScan::Started(stream, split_concurrency) => (stream, split_concurrency),
            NativeScan::Finished => return Ok(0),
        };

        match RUNTIME.block_on(stream.next()) {
            Some(partition) => {
                let partition = partition?;
                let handle = Box::into_raw(Box::new(NativePartition::Pending(
                    partition,
                    split_concurrency,
                ))) as jlong;
                ptr::write(slot, NativeScan::Started(stream, split_concurrency));
                Ok(handle)
            }
            None => Ok(0),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativePartition_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer == 0 {
        return;
    }
    drop(unsafe { Box::from_raw(pointer as *mut NativePartition) });
}

/// Write partition's estimated row count `[rows, cardinality]` into `out`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativePartition_rowCount(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    out: JLongArray,
) {
    try_or_throw(&mut env, |env| {
        let partition = unsafe { &*(pointer as *const NativePartition) };
        let NativePartition::Pending(partition, _) = partition else {
            throw_runtime!("row count unavailable: partition already started");
        };
        let (rows, cardinality) = match partition.row_count() {
            Precision::Exact(v) => (v as jlong, 2),
            Precision::Inexact(v) => (v as jlong, 1),
            Precision::Absent => (0, 0),
        };
        out.set_region(env, 0, &[rows, cardinality])?;
        Ok(())
    });
}

fn partition_into_arrow_stream(
    session: VortexSession,
    partition: PartitionRef,
    split_concurrency: Option<NonZeroUsize>,
) -> VortexResult<FFI_ArrowArrayStream> {
    let lookahead = split_concurrency
        .map_or(usize::MAX, NonZeroUsize::get)
        .min(POOL.worker_count().max(1));

    let array_stream = partition.execute()?;
    let dtype = array_stream.dtype().clone();

    let schema = Arc::new(session.arrow().to_arrow_schema(&dtype)?);
    let target = Arc::new(Field::new_struct("", schema.fields().clone(), false));

    let handle = RUNTIME.handle();
    let batches = array_stream
        .map(move |chunk| {
            let session = session.clone();
            let target = Arc::clone(&target);
            handle.spawn(async move {
                let chunk = chunk?;
                let mut ctx = session.create_execution_ctx();
                let arrow = chunk.execute_arrow(Some(target.as_ref()), &mut ctx)?;
                rebase_offsets(RecordBatch::from(arrow.as_struct().clone()))
            })
        })
        .buffered(lookahead);

    // Each JNI stream has a single consumer, so keep only one completed batch ahead of Java
    // rather than the per-core queue of `block_on_stream_thread_safe`.
    let iter =
        RUNTIME
            .block_on_stream_prefetch(batches)
            .map(|result: VortexResult<RecordBatch>| {
                result.map_err(|e| ArrowError::ExternalError(Box::new(e)))
            });

    let reader = RecordBatchIteratorAdapter::new(iter, schema);
    Ok(FFI_ArrowArrayStream::new(Box::new(reader)))
}

/// Consume a partition into the `FFI_ArrowArrayStream` pointed to by `stream_addr`. The
/// partition pointer is invalidated by this call; Java must not `free` it afterwards.
///
/// Ownership of `partition_ptr` is unconditionally transferred into this function — it
/// is boxed up front so that any subsequent error still drops the partition.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativePartition_scanArrow(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    partition_ptr: jlong,
    stream_addr: jlong,
) {
    // Take ownership first, so the partition is dropped even if later validation throws.
    debug_assert!(partition_ptr != 0, "null partition pointer");
    let partition = *unsafe { Box::from_raw(partition_ptr as *mut NativePartition) };

    try_or_throw(&mut env, |_| {
        if stream_addr == 0 {
            throw_runtime!("null arrow stream address");
        }

        let (partition, split_concurrency) = match partition {
            NativePartition::Pending(p, split_concurrency) => (p, split_concurrency),
            _ => throw_runtime!("partition already consumed"),
        };

        let session = unsafe { session_ref(session_ptr) }.clone();
        let arrow_stream = partition_into_arrow_stream(session, partition, split_concurrency)?;
        unsafe {
            ptr::write(stream_addr as *mut FFI_ArrowArrayStream, arrow_stream);
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchReader;
    use arrow_array::cast::AsArray;
    use arrow_array::ffi_stream::ArrowArrayStreamReader;
    use arrow_array::types::Int32Type;
    use futures::StreamExt;
    use futures::stream;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::array::stream::SendableArrayStream;
    use vortex::array::validity::Validity;
    use vortex::error::VortexResult;
    use vortex::expr::stats::Precision;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::scan::Partition;
    use vortex::scan::PartitionRef;
    use vortex::session::SessionExt;
    use vortex::session::SessionVar;
    use vortex::session::VortexSession;
    use vortex_arrow::ArrowSessionExt;

    use crate::RUNTIME;
    use crate::scan::partition_into_arrow_stream;

    #[derive(Debug)]
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Debug, Clone)]
    struct SessionLifetimeGuard(#[allow(dead_code)] Arc<DropCounter>);

    impl SessionVar for SessionLifetimeGuard {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    struct InMemoryPartition {
        chunks: Vec<StructArray>,
    }

    impl Partition for InMemoryPartition {
        fn as_any(&self) -> &(dyn Any + 'static) {
            self
        }

        fn index(&self) -> usize {
            0
        }

        fn row_count(&self) -> Precision<u64> {
            let total: usize = self.chunks.iter().map(StructArray::len).sum();
            Precision::Exact(total as u64)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            let dtype = self.chunks[0].dtype().clone();
            let iter = self.chunks.into_iter().map(|c| Ok(c.into_array()));
            Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream::iter(iter))))
        }
    }

    fn make_chunk(values: &[i32]) -> StructArray {
        StructArray::try_new(
            ["value"].into(),
            vec![PrimitiveArray::from_iter(values.iter().copied()).into_array()],
            values.len(),
            Validity::NonNullable,
        )
        .unwrap()
    }

    fn tracked_session() -> (VortexSession, Arc<AtomicUsize>) {
        let alive = Arc::new(AtomicUsize::new(1));
        let session = VortexSession::default().with_handle(RUNTIME.handle());
        drop(session.get_or_insert_with(|| {
            SessionLifetimeGuard(Arc::new(DropCounter(Arc::clone(&alive))))
        }));
        (session, alive)
    }

    #[test]
    fn arrow_export_retains_session_until_early_disposal() -> VortexResult<()> {
        let partition: PartitionRef = Box::new(InMemoryPartition {
            chunks: vec![make_chunk(&[10, 11, 12]), make_chunk(&[13, 14])],
        });
        let (session, alive) = tracked_session();
        let stream = partition_into_arrow_stream(session, partition, None)?;
        assert_eq!(
            alive.load(Ordering::SeqCst),
            1,
            "exported JNI Arrow stream must own a session handle before any poll"
        );
        drop(stream);
        // Drive canceled prefetch and conversion tasks on the shared current-thread runtime.
        for _ in 0..64 {
            if alive.load(Ordering::SeqCst) == 0 {
                break;
            }
            RUNTIME.block_on(RUNTIME.handle().spawn(async {}));
        }
        assert_eq!(
            alive.load(Ordering::SeqCst),
            0,
            "disposing the exported JNI Arrow stream must release its session handle"
        );
        Ok(())
    }

    #[test]
    fn arrow_export_retains_session_through_reads_and_disposal() -> VortexResult<()> {
        let chunk1 = make_chunk(&[10, 11, 12]);
        let chunk2 = make_chunk(&[13, 14]);
        let expected_dtype = chunk1.dtype().clone();
        let partition: PartitionRef = Box::new(InMemoryPartition {
            chunks: vec![chunk1, chunk2],
        });
        let (session, alive) = tracked_session();
        let expected_schema = session.arrow().to_arrow_schema(&expected_dtype)?;
        let stream = partition_into_arrow_stream(session, partition, None)?;
        assert_eq!(
            alive.load(Ordering::SeqCst),
            1,
            "exported JNI Arrow stream must own a session handle before any poll"
        );

        let reader = ArrowArrayStreamReader::try_new(stream)?;
        assert_eq!(reader.schema().as_ref(), &expected_schema);

        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>()?;
        let actual: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(actual, vec![10, 11, 12, 13, 14]);
        assert_eq!(
            alive.load(Ordering::SeqCst),
            0,
            "exhausting and dropping the JNI Arrow reader must release its session handle"
        );
        Ok(())
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    fn split_concurrency_bounds_completed_batches_ahead_of_the_consumer(#[case] max: usize) {
        let runtime = CurrentThreadRuntime::new();
        let pool = runtime.new_pool();
        pool.set_workers(2);
        let completed = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&completed);
        let handle = runtime.handle();
        let batches = stream::iter(0..16)
            .map(move |value| {
                let counter = Arc::clone(&counter);
                handle.spawn(async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    value
                })
            })
            .buffered(max);
        let mut iter = runtime.block_on_stream_prefetch(batches);
        assert_eq!(iter.next(), Some(0));

        // Let background work finish while the consumer keeps its first batch open.
        runtime.block_on(async {
            for _ in 0..64 {
                runtime.handle().spawn(async {}).await;
            }
        });
        assert!(
            completed.load(Ordering::SeqCst) <= max + 1,
            "completed batches escaped the conversion lookahead bound"
        );

        assert_eq!(iter.collect::<Vec<_>>(), (1..16).collect::<Vec<_>>());
    }
}
