// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::io::Write;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::TryStreamExt;
use tempfile::NamedTempFile;
use vortex::array::VortexSessionExecute;
use vortex::array::assert_arrays_eq;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;

use super::*;

fn view(value: &str) -> vx_view {
    vx_view {
        ptr: value.as_ptr().cast(),
        len: value.len(),
    }
}

fn names(values: &[&str]) -> VortexResult<FieldNames> {
    let views: Vec<_> = values.iter().map(|name| view(name)).collect();
    // SAFETY: All views and their string bytes remain live throughout parsing.
    unsafe { scan_columns(views.as_ptr(), views.len()) }
}

fn assert_error<T>(result: VortexResult<T>, message: &str) {
    let error = result.err().expect("expected an error");
    assert!(error.to_string().contains(message), "{error}");
}

#[test]
fn test_projection_names_are_owned_and_zero_count_means_all() -> VortexResult<()> {
    let parsed = {
        let name = String::from("値.x");
        names(&[&name, ""])?
    };
    assert_eq!(parsed, ["値.x", ""]);
    // SAFETY: Zero count ignores the pointer, including a null pointer.
    assert!(unsafe { scan_columns(ptr::null(), 0) }?.is_empty());
    let empty = vx_view {
        ptr: ptr::null(),
        len: 0,
    };
    // SAFETY: A null, zero-length view is a valid empty name.
    assert_eq!(unsafe { scan_columns(&raw const empty, 1) }?, [""]);
    Ok(())
}

#[test]
fn test_projection_wide_order_and_first_late_duplicate() -> VortexResult<()> {
    let columns: Vec<_> = (0..1024).rev().map(|i| format!("column_{i}")).collect();
    let mut projection: Vec<_> = columns.iter().map(String::as_str).collect();
    let parsed = names(&projection)?;
    assert_eq!(parsed, projection.as_slice());

    let duplicate = String::from("column_512");
    projection.extend([duplicate.as_str(), "column_1023"]);
    assert_error(
        names(&projection),
        "duplicate CUDA scan column: \"column_512\"",
    );
    Ok(())
}

#[test]
fn test_projection_rejects_invalid_names_and_counts() {
    let invalid_utf8 = vx_view {
        ptr: [0xffu8].as_ptr().cast(),
        len: 1,
    };
    let null_name = vx_view {
        ptr: ptr::null(),
        len: 1,
    };
    let long_name = vx_view {
        ptr: "x".as_ptr().cast(),
        len: usize::MAX,
    };
    let aligned = [view("x"), view("x")];
    let misaligned = aligned.as_ptr().cast::<u8>().wrapping_add(1).cast();
    for (columns, count, message) in [
        (ptr::null(), 1, "null CUDA scan columns"),
        (aligned.as_ptr(), usize::MAX, "column count is too large"),
        (misaligned, 1, "unaligned CUDA scan columns"),
        (&raw const invalid_utf8, 1, "invalid utf-8"),
        (&raw const null_name, 1, "null vx_view pointer"),
        (&raw const long_name, 1, "name is too long"),
        (aligned.as_ptr(), 2, "duplicate CUDA scan column: \"x\""),
    ] {
        // SAFETY: Invalid pointer/length combinations must be rejected before dereferencing;
        // all remaining views and bytes are live.
        assert_error(unsafe { scan_columns(columns, count) }, message);
    }
}

fn session() -> VortexSession {
    VortexSession::default().with_handle(ffi_runtime().handle())
}

fn table() -> VortexResult<StructArray> {
    StructArray::try_new(
        ["ids", "unused", "値.x"].into(),
        vec![
            PrimitiveArray::from_iter(0u32..5).into_array(),
            PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0]).into_array(),
            PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), None, Some(50)])
                .into_array(),
        ],
        5,
        Validity::NonNullable,
    )
}

fn file_bytes(
    session: &VortexSession,
    array: ArrayRef,
    cuda_block_rows: Option<usize>,
) -> VortexResult<ByteBuffer> {
    let strategy: Arc<dyn LayoutStrategy> = if let Some(block_rows) = cuda_block_rows {
        register_cuda_layout(session);
        cuda_write_strategy(session, block_rows)
    } else {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Arc::new(TableStrategy::new(Arc::clone(&flat), flat))
    };
    let mut bytes = ByteBufferMut::empty();
    ffi_runtime().block_on(
        session
            .write_options()
            .with_strategy(strategy)
            .write(&mut bytes, array.to_array_stream()),
    )?;
    Ok(bytes.freeze())
}

fn open_file(
    session: &VortexSession,
    array: ArrayRef,
    cuda_block_rows: Option<usize>,
) -> VortexResult<VortexFile> {
    session
        .open_options()
        .open_buffer(file_bytes(session, array, cuda_block_rows)?)
}

#[test]
fn test_cuda_write_strategy_preserves_high_cardinality_row_blocks() -> VortexResult<()> {
    let session = session();
    let unique = 70_000u32;
    let ids = PrimitiveArray::from_iter((0..unique).chain(0..unique)).into_array();
    let rows = ids.len();
    let input =
        StructArray::try_new(["ids"].into(), vec![ids], rows, Validity::NonNullable)?.into_array();
    let file = open_file(&session, input, Some(rows))?;
    let lengths: Vec<_> = ffi_runtime().block_on(
        projected_scan(&file, names(&["ids"])?, rows)?
            .into_array_stream()?
            .map_ok(|batch| batch.len())
            .try_collect(),
    )?;
    assert_eq!(lengths, [rows]);
    Ok(())
}

struct RejectSegments {
    inner: Arc<dyn SegmentSource>,
    forbidden: Vec<SegmentId>,
    rejected: AtomicUsize,
}

impl SegmentSource for RejectSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        if self.forbidden.contains(&id) {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Box::pin(async move {
                Err(vortex_err!("unselected column segment requested: {id}"))
            });
        }
        self.inner.request(id)
    }
}

#[test]
fn test_projection_cpu_never_requests_unselected_column_segments() -> VortexResult<()> {
    let session = session();
    let input = table()?;
    let columns = names(&["値.x", "ids"])?;
    let expected = input.project(columns.as_ref())?.into_array();
    let file = open_file(&session, input.into_array(), None)?;
    // TableStrategy writes one flat child per column, so child 1 is exactly the unused column.
    let children = file.footer().layout().children()?;
    let forbidden = children[1].segment_ids();
    assert!(!forbidden.is_empty());
    let source = Arc::new(RejectSegments {
        inner: file.segment_source(),
        forbidden,
        rejected: AtomicUsize::new(0),
    });
    let file = file.with_segment_source(Arc::<RejectSegments>::clone(&source));
    let actual = ffi_runtime().block_on(
        projected_scan(&file, columns, 2)?
            .into_array_stream()?
            .read_all(),
    )?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    assert_eq!(source.rejected.load(Ordering::Relaxed), 0);
    // The same reader must fail without projection, proving the guard actually observes reads.
    assert_error(
        ffi_runtime().block_on(
            projected_scan(&file, names(&[])?, 0)?
                .into_array_stream()?
                .read_all(),
        ),
        "unselected column segment requested",
    );
    assert!(source.rejected.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn test_projection_ffi_validation_without_cuda() {
    let mut stream = ArrowDeviceArrayStream {
        device_type: -1,
        get_schema: None,
        get_next: None,
        get_last_error: None,
        release: None,
        private_data: ptr::null_mut(),
    };
    let mut error = ptr::null_mut();
    // SAFETY: Output pointers are writable; invalid columns must fail before session/path use.
    let status = unsafe {
        vx_cuda_scan_path_arrow_device_stream_projected(
            ptr::null(),
            view(""),
            ptr::null(),
            ptr::null(),
            1,
            &raw mut stream,
            &raw mut error,
        )
    };
    assert_eq!(status, VX_CUDA_ERR);
    assert_eq!(stream.device_type, -1);
    assert!(stream.release.is_none());
    assert!(!error.is_null());
    // SAFETY: This call owns the returned error and frees it exactly once.
    unsafe { vortex_ffi::vx_error_free(error) };
    // SAFETY: Null output is rejected before any other input is used; error output is optional.
    assert_eq!(
        unsafe {
            vx_cuda_scan_path_arrow_device_stream_projected(
                ptr::null(),
                view(""),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        VX_CUDA_ERR
    );
}

fn stream_error(stream: &mut ArrowDeviceArrayStream) -> String {
    // SAFETY: The callback and returned C string belong to this live stream.
    unsafe {
        stream
            .get_last_error
            .and_then(|callback| callback(stream).as_ref())
            .map(|message| CStr::from_ptr(message).to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

fn open_stream(
    session: &VortexSession,
    path: &str,
    options: &vx_cuda_scan_options,
) -> ArrowDeviceArrayStream {
    let mut output = MaybeUninit::<ArrowDeviceArrayStream>::uninit();
    let mut error = ptr::null_mut();
    let handle = test_session(session.clone());
    let columns = [view("値.x"), view("ids")];
    // SAFETY: All borrowed inputs and writable outputs are live for this call.
    let status = unsafe {
        vx_cuda_scan_path_arrow_device_stream_projected(
            handle,
            view(path),
            options,
            columns.as_ptr(),
            columns.len(),
            output.as_mut_ptr(),
            &raw mut error,
        )
    };
    // SAFETY: This is the sole release of the borrowed session handle.
    unsafe { free_test_session(handle) };
    assert_eq!(status, VX_CUDA_OK);
    assert!(error.is_null());
    // SAFETY: A successful call initialized the stream, which owns its session state.
    unsafe { output.assume_init() }
}

pub(super) fn stream_schema(stream: &mut ArrowDeviceArrayStream) -> FFI_ArrowSchema {
    let mut schema = FFI_ArrowSchema::empty();
    let get_schema = stream.get_schema.expect("missing get_schema");
    // SAFETY: This live stream owns the callback; schema is writable.
    assert_eq!(
        unsafe { get_schema(stream, (&raw mut schema).cast()) },
        0,
        "{}",
        stream_error(stream)
    );
    schema
}

fn batch_lengths(stream: &mut ArrowDeviceArrayStream) -> Vec<i64> {
    let get_next = stream.get_next.expect("missing get_next");
    let mut lengths = Vec::new();
    loop {
        let mut array = empty_device_array();
        // SAFETY: This live stream owns the callback; array is writable.
        assert_eq!(
            unsafe { get_next(stream, &raw mut array) },
            0,
            "{}",
            stream_error(stream)
        );
        if array.array.release.is_none() {
            break;
        }
        assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(array.array.n_children, 2);
        lengths.push(array.array.length);
        release_device_array(&mut array);
    }
    lengths
}

#[cuda_test]
fn test_projection_gpu_local_file_schema_and_batch_boundaries() -> VortexResult<()> {
    for (block_rows, batch_rows) in [(0, 2), (2, 3)] {
        let session = session().with_some(CudaSession::try_default()?);
        let mut file = NamedTempFile::new()?;
        file.write_all(&file_bytes(
            &session,
            table()?.into_array(),
            Some(block_rows),
        )?)?;
        let path = file
            .path()
            .to_str()
            .ok_or_else(|| vortex_err!("non-UTF-8 test path"))?;
        let options = vx_cuda_scan_options {
            batch_rows,
            ..Default::default()
        };
        let mut stream = open_stream(&session, path, &options);
        // The stream must retain its session state after the caller releases its session.
        drop(session);
        let mut schema = stream_schema(&mut stream);
        let expected_fields = vec![
            Field::new("値.x", DataType::Int64, true),
            Field::new("ids", DataType::UInt32, false),
        ];
        assert_eq!(Schema::try_from(&schema)?, Schema::new(expected_fields));
        assert_eq!(batch_lengths(&mut stream), [2, 2, 1]);
        let release = stream.release.expect("missing release");
        // SAFETY: Both objects are live and released exactly once.
        unsafe {
            release_schema(&mut schema);
            release(&raw mut stream);
        }
    }
    Ok(())
}
