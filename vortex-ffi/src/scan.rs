// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(non_camel_case_types)]
#![deny(missing_docs)]

use core::slice;
use std::ffi::c_int;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use futures::StreamExt;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::expr::stats::Precision;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::expr::root;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::scan::DataSourceScan;
use vortex::scan::Partition;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::data_source::vx_data_source;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::expression::vx_expression;
use crate::session::vx_session;

pub enum VxScan {
    Pending(Box<dyn DataSourceScan>),
    Started(PartitionStream),
    Finished,
}
crate::box_wrapper!(
    /// A scan is a single traversal of a data source with projections and
    /// filters. A scan can be consumed only once.
    VxScan,
    vx_scan);

pub enum VxPartitionScan {
    Pending(Box<dyn Partition>),
    Started(SendableArrayStream),
    Finished,
}
crate::box_wrapper!(
    /// A partition is an independent unit of work. Call vx_partition_next repeatedly to
    /// retrieve arrays, then free the partition with vx_partition_free.
    VxPartitionScan,
    vx_partition);

// We parse Selection from vx_scan_selection[_include], so we don't need
// to instantiate VX_SELECTION_* items directly.
#[repr(C)]
#[allow(dead_code)]
#[cfg_attr(test, derive(Default))]
pub enum vx_scan_selection_include {
    #[cfg_attr(test, default)]
    VX_SELECTION_INCLUDE_ALL = 0,
    /// Include rows at the indices in vx_scan_selection.idx.
    VX_SELECTION_INCLUDE_RANGE = 1,
    /// Exclude rows at the indices in vx_scan_selection.idx.
    VX_SELECTION_EXCLUDE_RANGE = 2,
}

/// Scan row selection.
/// "idx" is copied while calling vx_data_source_scan and can be freed after.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_selection {
    /// Used only when "include" is not VX_SELECTION_INCLUDE_ALL.
    /// If set, must point to an array of len "idx_len" row_indices.
    pub idx: *const u64,
    /// Used only when "include" is not VX_SELECTION_INCLUDE_ALL
    pub idx_len: usize,
    pub include: vx_scan_selection_include,
}

/// Scan options. All fields are optional. To return everything,
/// zero-initialize this struct.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_options {
    /// What columns to return. NULL means all columns.
    pub projection: *const vx_expression,
    /// Predicate expression. NULL means no filter.
    pub filter: *const vx_expression,
    /// Row range [begin, end). Setting row_range_begin and row_range_end to 0
    /// means no limit.
    pub row_range_begin: u64,
    pub row_range_end: u64,
    /// Row-index filter applied after row_range.
    pub selection: vx_scan_selection,
    /// Maximum number of rows to return. 0 means no limit.
    pub limit: u64,
    /// Upper limit for parallelism. 0 means no limit.
    /// Scan will return at most "max_threads" partitions.
    pub max_threads: u64,
    /// If true, return in storage order.
    pub ordered: bool,
}

#[repr(C)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq, Default))]
pub enum vx_estimate_type {
    /// No estimate is available.
    #[cfg_attr(test, default)]
    VX_ESTIMATE_UNKNOWN = 0,
    /// The value in vx_estimate.estimate is exact.
    VX_ESTIMATE_EXACT = 1,
    /// The value in vx_estimate.estimate is an upper bound.
    VX_ESTIMATE_INEXACT = 2,
}

/// Used for estimating number of partitions in a data source or number of rows
/// in a partition.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_estimate {
    pub r#type: vx_estimate_type,
    /// Set only when "type" is not VX_ESTIMATE_UNKNOWN.
    pub estimate: u64,
}

fn scan_request(opts: *const vx_scan_options) -> VortexResult<ScanRequest> {
    if opts.is_null() {
        return Ok(ScanRequest::default());
    }
    let opts = unsafe { &*opts };

    let projection = if opts.projection.is_null() {
        root()
    } else {
        vx_expression::as_ref(opts.projection).clone()
    };

    let filter = if opts.filter.is_null() {
        None
    } else {
        Some(vx_expression::as_ref(opts.filter).clone())
    };

    let selection = &opts.selection;
    let selection = match selection.include {
        vx_scan_selection_include::VX_SELECTION_INCLUDE_ALL => Selection::All,
        vx_scan_selection_include::VX_SELECTION_INCLUDE_RANGE => {
            vortex_ensure!(!selection.idx.is_null());
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::IncludeByIndex(buf)
        }
        vx_scan_selection_include::VX_SELECTION_EXCLUDE_RANGE => {
            vortex_ensure!(!selection.idx.is_null());
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::ExcludeByIndex(buf)
        }
    };

    let ordered = opts.ordered;

    let start = opts.row_range_begin;
    let end = opts.row_range_end;
    let row_range = (start > 0 || end > 0).then_some(Range { start, end });

    let limit = (opts.limit != 0).then_some(opts.limit);

    Ok(ScanRequest {
        projection,
        filter,
        row_range,
        selection,
        ordered,
        limit,
    })
}

fn write_estimate<T: Into<u64>>(estimate: Option<Precision<T>>, out: &mut vx_estimate) {
    match estimate {
        Some(Precision::Exact(value)) => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_EXACT;
            out.estimate = value.into();
        }
        Some(Precision::Inexact(value)) => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_INEXACT;
            out.estimate = value.into();
        }
        None => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_UNKNOWN;
        }
    }
}

/// Scan a data source.
///
/// Return an owned scan that must be freed with vx_scan_free. A scan may be
/// consumed only once.
///
/// "options" and "estimate" may be NULL.
///
/// If "options" is NULL, all rows and columns are returned.
/// If "estimate" is not NULL, the estimated partition count is written to
/// *estimate before returning.
///
/// Returns NULL and writes an error to "*err" on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_scan(
    data_source: *const vx_data_source,
    options: *const vx_scan_options,
    estimate: *mut vx_estimate,
    err: *mut *mut vx_error,
) -> *mut vx_scan {
    try_or(err, ptr::null_mut(), || {
        let request = scan_request(options)?;
        RUNTIME.block_on(async {
            let scan = vx_data_source::as_ref(data_source).scan(request).await?;
            if !estimate.is_null() {
                write_estimate(
                    scan.partition_count().map(|x| match x {
                        Precision::Exact(v) => Precision::Exact(v as u64),
                        Precision::Inexact(v) => Precision::Inexact(v as u64),
                    }),
                    unsafe { &mut *estimate },
                );
            }
            Ok(vx_scan::new(VxScan::Pending(scan)))
        })
    })
}

/// Return borrowed vx_scan's dtype.
/// This function will fail if called after vx_scan_next_partition.
/// Called must not free the returned pointer as its lifetime is bound to the
/// lifetime of the scan.
/// On error returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_dtype(
    scan: *const vx_scan,
    err: *mut *mut vx_error,
) -> *const vx_dtype {
    try_or(err, ptr::null(), || {
        let scan = vx_scan::as_ref(scan);
        let VxScan::Pending(scan) = scan else {
            vortex_bail!("dtype unavailable: scan already started");
        };
        Ok(vx_dtype::new_ref(scan.dtype()))
    })
}

/// Return an owned partition from a scan.
/// The returned partition must be freed with vx_partition_free.
///
/// On success returns a partition.
/// On exhaustion (no more partitions in scan) returns NULL but doesn't set
/// "err".
/// On error returns NULL and sets "err".
///
/// This function is thread-unsafe. Callers running a multi-threaded pipeline
/// should synchronise on calls to this function and dispatch each produced
/// partition to a dedicated worker thread.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_next_partition(
    scan: *mut vx_scan,
    err: *mut *mut vx_error,
) -> *mut vx_partition {
    let scan = vx_scan::as_mut(scan);
    let scan = &mut *scan;
    unsafe {
        let ptr = scan as *mut VxScan;

        let on_finish = || -> VortexResult<*mut vx_partition> {
            ptr::write(ptr, VxScan::Finished);
            Ok(ptr::null_mut())
        };

        let on_stream = |mut stream: PartitionStream| -> VortexResult<*mut vx_partition> {
            match RUNTIME.block_on(stream.next()) {
                Some(partition) => {
                    let partition = VxPartitionScan::Pending(partition?);
                    let partition = vx_partition::new(partition);
                    ptr::write(ptr, VxScan::Started(stream));
                    Ok(partition)
                }
                None => on_finish(),
            }
        };

        let owned = ptr::read(ptr);
        try_or_default(err, || match owned {
            VxScan::Pending(scan) => on_stream(scan.partitions()),
            VxScan::Started(stream) => on_stream(stream),
            VxScan::Finished => on_finish(),
        })
    }
}

/// Get partition's estimated row count.
/// Must be called before the first call to vx_partition_next.
///
/// On success, returns 0.
/// On error, return 1 and sets "error".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_row_count(
    partition: *const vx_partition,
    count: *mut vx_estimate,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let partition = vx_partition::as_ref(partition);
        let VxPartitionScan::Pending(partition) = partition else {
            vortex_bail!("row count unavailable: partition already started");
        };
        write_estimate(partition.row_count(), unsafe { &mut *count });
        Ok(0)
    })
}

/// Scan partition to ArrowArrayStream.
/// Consumes partition fully: subsequent calls to vx_partition_scan_arrow or
/// vx_partition_next are undefined behaviour.
/// This call blocks current thread until underlying stream is fully consumed.
///
/// Caller must not free partition after calling this function.
///
/// On success, sets "stream" and returns 0.
/// On error, sets "err" and returns 1, freeing the partition.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_scan_arrow(
    session: *const vx_session,
    partition: *mut vx_partition,
    stream: *mut FFI_ArrowArrayStream,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let partition = match *vx_partition::into_box(partition) {
            VxPartitionScan::Pending(partition) => partition,
            _ => vortex_bail!(
                "Can't consume partition into ArrowArrayStream: partition already being consumed"
            ),
        };
        let array_stream = partition.execute()?;
        let dtype = array_stream.dtype();

        let schema = dtype.to_arrow_schema()?;
        let schema = Arc::new(schema);
        let data_type = DataType::Struct(schema.fields().clone());

        let session = vx_session::as_ref(session);

        let on_chunk = move |chunk: VortexResult<ArrayRef>| -> VortexResult<RecordBatch> {
            let chunk: ArrayRef = chunk?;
            let mut ctx: ExecutionCtx = session.create_execution_ctx();
            let arrow = chunk.execute_arrow(Some(&data_type), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        };

        let iter = RUNTIME
            .block_on_stream(array_stream)
            .map(on_chunk)
            .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        let reader = RecordBatchIteratorAdapter::new(iter, schema);
        let arrow_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        unsafe {
            ptr::write(stream, arrow_stream);
        };
        Ok(0)
    })
}

/// Return an owned owned array from a partition.
/// The returned array must be freed with vx_array_free.
///
/// On success returns an array.
/// On exhaustion (no more arrays in partition) returns NULL but doesn't set
/// "err".
/// On error return NULL and sets "err".
///
/// This function is not thread-safe: call from one thread per partition.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_next(
    partition: *mut vx_partition,
    err: *mut *mut vx_error,
) -> *const vx_array {
    let partition = vx_partition::as_mut(partition);
    unsafe {
        let ptr = partition as *mut VxPartitionScan;

        let on_finish = || -> VortexResult<*const vx_array> {
            ptr::write(ptr, VxPartitionScan::Finished);
            Ok(ptr::null_mut())
        };

        let on_stream = |mut stream: SendableArrayStream| -> VortexResult<*const vx_array> {
            match RUNTIME.block_on(stream.next()) {
                Some(array) => {
                    let array = vx_array::new(Arc::new(array?));
                    ptr::write(ptr, VxPartitionScan::Started(stream));
                    Ok(array)
                }
                None => on_finish(),
            }
        };

        let owned = ptr::read(ptr);
        try_or_default(err, || match owned {
            VxPartitionScan::Pending(partition) => on_stream(partition.execute()?),
            VxPartitionScan::Started(stream) => on_stream(stream),
            VxPartitionScan::Finished => on_finish(),
        })
    }
}

// Object store error: Generic LocalFileSystem error: Unable to convert
// URL "file:///C:%255CWindows%255CSystemTemp%255C.tmpRXzX38" to filesystem path
// https://github.com/servo/rust-url/issues/1077
#[cfg(not(windows))]
#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use vortex::VortexSessionDefault;
    use vortex::array::arrays::StructArray;
    use vortex::session::VortexSession;
    use vortex_array::ExecutionCtx;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::expression::vx_binary_operator;
    use crate::expression::vx_expression_binary;
    use crate::expression::vx_expression_free;
    use crate::expression::vx_expression_get_item;
    use crate::expression::vx_expression_literal;
    use crate::expression::vx_expression_root;
    use crate::scalar::vx_scalar_free;
    use crate::scalar::vx_scalar_new_u64;
    use crate::scan::vx_data_source_scan;
    use crate::scan::vx_estimate;
    use crate::scan::vx_partition_free;
    use crate::scan::vx_partition_next;
    use crate::scan::vx_partition_row_count;
    use crate::scan::vx_scan_free;
    use crate::scan::vx_scan_next_partition;
    use crate::scan::vx_scan_options;
    use crate::scan::vx_scan_selection;
    use crate::scan::vx_scan_selection_include;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::tests::SAMPLE_ROWS;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    /// Perform a scan with options over a sample file, return owned read array and
    /// original generated array for the sample file.
    fn scan(options: *const vx_scan_options) -> (*const vx_array, StructArray) {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);
            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let ds_options = vx_data_source_options {
                paths: path.as_ptr(),
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let mut error = ptr::null_mut();
            let scan = vx_data_source_scan(ds, options, ptr::null_mut(), &raw mut error);
            assert_no_error(error);
            assert!(!scan.is_null());

            let partition = vx_scan_next_partition(scan, &raw mut error);
            assert_no_error(error);
            assert!(!partition.is_null());

            let array = vx_partition_next(partition, &raw mut error);
            assert_no_error(error);
            assert!(!array.is_null());

            assert!(vx_partition_next(partition, &raw mut error).is_null());
            assert_no_error(error);
            assert!(vx_partition_next(partition, &raw mut error).is_null());
            assert_no_error(error);

            vx_partition_free(partition);
            vx_scan_free(scan);
            vx_data_source_free(ds);
            vx_session_free(session);

            (array, struct_array)
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_no_options() {
        let (array, struct_array) = scan(ptr::null());
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_all() {
        let opts = vx_scan_options::default();
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_single_field() {
        unsafe {
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            for (field, c_field) in [("age", c"age"), ("height", c"height"), ("name", c"name")] {
                let field_expr = vx_expression_get_item(c_field.as_ptr(), root);
                assert!(!field_expr.is_null());
                opts.projection = field_expr;
                let (array, struct_array) = scan(&raw const opts);
                assert_arrays_eq!(
                    vx_array::as_ref(array),
                    struct_array.unmasked_field_by_name(field).unwrap()
                );
                vx_array_free(array);
                vx_expression_free(field_expr);
            }
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_sum() {
        let session = VortexSession::default();
        let mut ctx = ExecutionCtx::new(session);
        unsafe {
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            let expr_age = vx_expression_get_item(c"age".as_ptr(), root);
            let expr_height = vx_expression_get_item(c"height".as_ptr(), root);
            let expr_sum =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_ADD, expr_age, expr_height);

            opts.projection = expr_sum;
            let (array, _) = scan(&raw const opts);
            {
                let array = vx_array::as_ref(array);
                let stats = array.statistics();
                assert!(stats.compute_is_sorted(&mut ctx).unwrap());
                assert_eq!(stats.compute_min(&mut ctx), Some(0));
                assert_eq!(
                    stats.compute_max(&mut ctx),
                    Some(200 * (SAMPLE_ROWS - 1) + 199)
                );
            }
            vx_array_free(array);

            vx_expression_free(expr_age);
            vx_expression_free(expr_height);
            vx_expression_free(expr_sum);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(c"age".as_ptr(), root);
            let value = vx_scalar_new_u64(100, false);
            let mut error = ptr::null_mut();
            let lit_100 = vx_expression_literal(value, &raw mut error);
            assert_no_error(error);
            vx_scalar_free(value);
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);

            let opts = vx_scan_options {
                filter,
                ..Default::default()
            };
            let (array, _) = scan(&raw const opts);
            assert_eq!(vx_array::as_ref(array).len(), 100);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_project() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(c"age".as_ptr(), root);
            let value = vx_scalar_new_u64(100, false);
            let mut error = ptr::null_mut();
            let lit_100 = vx_expression_literal(value, &raw mut error);
            assert_no_error(error);
            vx_scalar_free(value);
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);
            let projection = vx_expression_get_item(c"age".as_ptr(), root);

            let opts = vx_scan_options {
                projection,
                filter,
                ..Default::default()
            };
            let (array, _) = scan(&raw const opts);
            assert_eq!(vx_array::as_ref(array).len(), 100);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(projection);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_range() {
        let opts = vx_scan_options {
            row_range_begin: 50,
            row_range_end: 100,
            ..Default::default()
        };
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), 50);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_selection() {
        let indices = [0u64, 50, 100, 150, 199];
        let opts = vx_scan_options {
            selection: vx_scan_selection {
                idx: indices.as_ptr(),
                idx_len: indices.len(),
                include: vx_scan_selection_include::VX_SELECTION_INCLUDE_RANGE,
            },
            ..Default::default()
        };
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), indices.len());
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_limit() {
        let opts = vx_scan_options {
            limit: 50,
            ..Default::default()
        };
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), 50);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_ordered() {
        let opts = vx_scan_options {
            ordered: true,
            ..Default::default()
        };
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_count() {
        unsafe {
            let session = vx_session_new();
            let (sample, _) = write_sample(session);
            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let ds_options = vx_data_source_options {
                paths: path.as_ptr(),
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);

            let mut error = ptr::null_mut();
            let scan_ptr = vx_data_source_scan(ds, ptr::null(), ptr::null_mut(), &raw mut error);
            assert_no_error(error);

            let mut error = ptr::null_mut();
            let partition = vx_scan_next_partition(scan_ptr, &raw mut error);
            assert_no_error(error);
            assert!(!partition.is_null());

            let mut count: vx_estimate = std::mem::zeroed();
            let result = vx_partition_row_count(partition, &raw mut count, &raw mut error);
            assert_no_error(error);
            assert_eq!(result, 0);

            vx_partition_free(partition);
            vx_scan_free(scan_ptr);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
