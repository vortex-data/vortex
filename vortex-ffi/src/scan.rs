#![allow(non_camel_case_types)]

use core::slice;
use std::ffi::c_int;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use futures::StreamExt;
use vortex::array::expr::stats::Precision;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::Selection;
use vortex::scan::api::DataSourceScan;
use vortex::scan::api::Partition;
use vortex::scan::api::PartitionStream;
use vortex::scan::api::ScanRequest;

use crate::error::vx_error;
use crate::RUNTIME;
use crate::array::vx_array;
use crate::data_source::vx_data_source;
use crate::error::try_or_default;
use crate::error::write_error;
use crate::expression::vx_expression;

pub enum VxScanState {
    Pending(Box<dyn DataSourceScan>),
    Started(PartitionStream),
    Finished,
}
pub type VxScan = Mutex<VxScanState>;
crate::box_wrapper!(VxScan, vx_scan);

pub enum VxPartitionScan {
    Pending(Box<dyn Partition>),
    Started(SendableArrayStream),
    Finished,
}
crate::box_wrapper!(
    /// A partition is a contiguous chunk of memory from which you can
    /// interatively get vx_arrays.
    /// TODO We're going away from exposing partitions to user, revise
    /// design
    VxPartitionScan,
    vx_partition);

#[repr(C)]
pub enum vx_scan_selection_include {
    VX_S_INCLUDE_ALL = 0,
    VX_S_INCLUDE_RANGE = 1,
    VX_S_EXCLUDE_RANGE = 2,
}

#[repr(C)]
pub struct vx_scan_selection {
    pub idx: *mut u64,
    pub idx_len: usize,
    pub include: vx_scan_selection_include,
}

// Distinct from ScanRequest for easier option handling from C
#[repr(C)]
pub struct vx_scan_options {
    pub projection: *const vx_expression,
    pub filter: *const vx_expression,
    pub row_range_begin: u64,
    pub row_range_end: u64,
    pub selection: vx_scan_selection,
    pub limit: u64,
    pub ordered: c_int,
}

#[repr(C)]
pub enum vx_estimate_boundary {
    VX_ESTIMATE_UNKNOWN = 0,
    VX_ESTIMATE_EXACT = 1,
    VX_ESTIMATE_INEXACT = 2,
}

#[repr(C)]
pub struct vx_estimate {
    estimate: u64,
    boundary: vx_estimate_boundary,
}

fn scan_request(opts: *const vx_scan_options) -> VortexResult<ScanRequest> {
    if opts.is_null() {
        return Ok(ScanRequest::default());
    }
    let opts = unsafe { &*opts };

    let projection = if opts.projection.is_null() {
        vortex_bail!("empty opts.projection");
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
        vx_scan_selection_include::VX_S_INCLUDE_ALL => Selection::All,
        vx_scan_selection_include::VX_S_INCLUDE_RANGE => {
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::IncludeByIndex(buf)
        }
        vx_scan_selection_include::VX_S_EXCLUDE_RANGE => {
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::ExcludeByIndex(buf)
        }
    };

    let ordered = opts.ordered == 1;

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

#[unsafe(no_mangle)]
// Create a new owned data source scan which must be freed by the caller.
// Scan can be consumed only once.
// Returns NULL and sets err on error.
// options may not be NULL.
pub unsafe extern "C-unwind" fn vx_data_source_scan(
    data_source: *const vx_data_source,
    options: *const vx_scan_options,
    err: *mut *mut vx_error,
) -> *mut vx_scan {
    try_or_default(err, || {
        let request = scan_request(options)?;
        RUNTIME.block_on(async {
            let scan = vx_data_source::as_ref(data_source).scan(request).await?;
            Ok(vx_scan::new(Box::new(Mutex::new(VxScanState::Pending(scan)))))
        })
    })
}

fn estimate<T: Into<u64>>(estimate: Option<Precision<T>>, out: &mut vx_estimate) {
    match estimate {
        Some(Precision::Exact(value)) => {
            out.boundary = vx_estimate_boundary::VX_ESTIMATE_EXACT;
            out.estimate = value.into();
        }
        Some(Precision::Inexact(value)) => {
            out.boundary = vx_estimate_boundary::VX_ESTIMATE_INEXACT;
            out.estimate = value.into();
        }
        None => {
            out.boundary = vx_estimate_boundary::VX_ESTIMATE_UNKNOWN;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_partition_count(
    scan: *const vx_scan,
    count: *mut vx_estimate,
    err: *mut *mut vx_error,
) {
    let count = unsafe { &mut *count };
    let scan = vx_scan::as_ref(scan);
    let mut scan = scan.lock().expect("failed to lock mutex");
    let scan = &mut *scan;
    let VxScanState::Pending(scan) = scan else {
        write_error(
            err,
            "can't get partition count of a scan that's already started",
        );
        return;
    };
    estimate(scan.partition_count().map(|x| match x {
        Precision::Exact(v) => Precision::Exact(v as u64),
        Precision::Inexact(v) => Precision::Inexact(v as u64),
    }), count)
}

#[unsafe(no_mangle)]
/// Get next owned partition out of a scan request.
/// Caller must free this partition using vx_partition_free.
/// This method is thread-safe.
/// If using in a sync multi-thread runtime, users are encouraged to create a
/// worker thread per partition.
/// Returns NULL and doesn't set err on exhaustion.
/// Returns NULL and sets err on error.
pub unsafe extern "C-unwind" fn vx_scan_next(
    scan: *mut vx_scan,
    err: *mut *mut vx_error,
) -> *mut vx_partition {
    let scan = vx_scan::as_mut(scan);
    let mut scan = scan.lock().expect("failed to lock mutex");
    let scan = &mut *scan;
    unsafe {
        let ptr = scan as *mut VxScanState;

        let on_finish = || -> VortexResult<*mut vx_partition> {
            ptr::write(ptr, VxScanState::Finished);
            Ok(ptr::null_mut())
        };

        let on_stream = |mut stream: PartitionStream| -> VortexResult<*mut vx_partition> {
            match RUNTIME.block_on(stream.next()) {
                Some(partition) => {
                    let partition = VxPartitionScan::Pending(partition?);
                    let partition = vx_partition::new(Box::new(partition));
                    ptr::write(ptr, VxScanState::Started(stream));
                    Ok(partition)
                }
                None => on_finish(),
            }
        };

        let owned = ptr::read(ptr);
        try_or_default(err, || match owned {
            VxScanState::Pending(scan) => on_stream(scan.partitions()),
            VxScanState::Started(stream) => on_stream(stream),
            VxScanState::Finished => on_finish(),
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_row_count(
    partition: *const vx_partition,
    count: *mut vx_estimate,
    err: *mut *mut vx_error
) {
    let partition = vx_partition::as_ref(partition);
    let VxPartitionScan::Pending(partition) = partition else {
        write_error(
            err,
            "can't get row count of a partition that's already started",
        );
        return;
    };
    estimate(partition.row_count(), unsafe { &mut *count} )
}

// TODO export nanoarrow headers?

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_scan_arrow(
    _partition: *const vx_partition,
    _stream: *mut FFI_ArrowArrayStream,
    err: *mut *mut vx_error,
) {
    write_error(err, "failed to scan partition to Arrow");
}

#[unsafe(no_mangle)]
/// Get next vx_array out of this partition.
/// Thread-unsafe.
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

#[unsafe(no_mangle)]
/// Scan progress between 0.0 and 1.0
pub unsafe extern "C-unwind" fn vx_scan_progress(_scan: *const vx_scan) -> f64 {
    0.0
}
