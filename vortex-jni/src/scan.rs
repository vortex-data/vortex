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

use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::sys::jboolean;
use jni::sys::jlong;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
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

use crate::RUNTIME;
use crate::data_source::NativeDataSource;
use crate::dtype::strip_views;
use crate::errors::try_or_throw;
use crate::session::session_ref;

/// Opaque scan handle. Holds a three-state machine: either the scan is pending (not yet
/// realized as a stream), actively streaming partitions, or finished.
#[allow(dead_code)]
pub(crate) enum NativeScan {
    Pending(Box<dyn DataSourceScan>),
    Started(PartitionStream),
    Finished,
}

/// Opaque partition handle with the same three-state machine shape.
#[allow(dead_code)]
pub(crate) enum NativePartition {
    Pending(PartitionRef),
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
    selection_include: u8,
    limit: jlong,
    ordered: jboolean,
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
        1 => Selection::IncludeByIndex(Buffer::copy_from(selection_idx)),
        2 => Selection::ExcludeByIndex(Buffer::copy_from(selection_idx)),
        other => vortex_bail!("unknown selection include code: {other}"),
    };

    let row_range = (row_range_begin > 0 || row_range_end > 0).then_some(Range {
        start: row_range_begin as u64,
        end: row_range_end as u64,
    });

    let limit = (limit > 0).then_some(limit as u64);

    Ok(ScanRequest {
        projection,
        filter,
        row_range,
        selection,
        ordered,
        limit,
    })
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
    selection_include: jni::sys::jbyte,
    limit: jlong,
    ordered: jboolean,
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

        let request = build_scan_request(
            projection_ptr,
            filter_ptr,
            row_range_begin,
            row_range_end,
            &selection_idx,
            selection_include as u8,
            limit,
            ordered,
        )?;

        let scan = RUNTIME.block_on(async { ds.inner().scan(request).await })?;
        Ok(Box::into_raw(Box::new(NativeScan::Pending(scan))) as jlong)
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
    pointer: jlong,
    schema_addr: jlong,
) {
    try_or_throw(&mut env, |_| {
        if schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let scan = unsafe { &*(pointer as *const NativeScan) };
        let NativeScan::Pending(scan) = scan else {
            throw_runtime!("schema unavailable: scan already started");
        };
        crate::dtype::export_dtype_to_arrow(scan.dtype(), schema_addr)?;
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
        let NativeScan::Pending(scan) = scan else {
            throw_runtime!("partition count unavailable: scan already started");
        };
        let (rows, cardinality) = match scan.partition_count() {
            Some(Precision::Exact(v)) => (v as jlong, 2),
            Some(Precision::Inexact(v)) => (v as jlong, 1),
            None => (0, 0),
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

        let mut stream = match owned {
            NativeScan::Pending(scan) => scan.partitions(),
            NativeScan::Started(stream) => stream,
            NativeScan::Finished => return Ok(0),
        };

        match RUNTIME.block_on(stream.next()) {
            Some(partition) => {
                let partition = partition?;
                let handle = Box::into_raw(Box::new(NativePartition::Pending(partition))) as jlong;
                ptr::write(slot, NativeScan::Started(stream));
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
        let NativePartition::Pending(partition) = partition else {
            throw_runtime!("row count unavailable: partition already started");
        };
        let (rows, cardinality) = match partition.row_count() {
            Some(Precision::Exact(v)) => (v as jlong, 2),
            Some(Precision::Inexact(v)) => (v as jlong, 1),
            None => (0, 0),
        };
        out.set_region(env, 0, &[rows, cardinality])?;
        Ok(())
    });
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

        let partition = match partition {
            NativePartition::Pending(p) => p,
            _ => throw_runtime!("partition already consumed"),
        };

        let array_stream = partition.execute()?;
        let dtype = array_stream.dtype().clone();

        let raw_schema = dtype.to_arrow_schema()?;
        let viewless = strip_views(DataType::Struct(raw_schema.fields().clone()));
        let fields = match viewless {
            DataType::Struct(fields) => fields,
            _ => unreachable!("Vortex DType always exports as a struct"),
        };
        let schema = Arc::new(arrow_schema::Schema::new(fields));
        let data_type = DataType::Struct(schema.fields().clone());

        let session = unsafe { session_ref(session_ptr) };

        let iter = RUNTIME
            .block_on_stream_thread_safe(|_handle| array_stream)
            .map(
                move |chunk: VortexResult<ArrayRef>| -> VortexResult<RecordBatch> {
                    let chunk: ArrayRef = chunk?;
                    let mut ctx: ExecutionCtx = session.create_execution_ctx();
                    let arrow = chunk.execute_arrow(Some(&data_type), &mut ctx)?;
                    Ok(RecordBatch::from(arrow.as_struct().clone()))
                },
            )
            .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        let reader = RecordBatchIteratorAdapter::new(iter, schema);
        let arrow_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        unsafe {
            ptr::write(stream_addr as *mut FFI_ArrowArrayStream, arrow_stream);
        }
        Ok(())
    });
}
