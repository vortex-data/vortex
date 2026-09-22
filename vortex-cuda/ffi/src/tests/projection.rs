// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use cudarc::driver::CudaContext;
use cudarc::driver::result;
use cudarc::driver::sys::CUevent;
use futures::TryStreamExt;
use tempfile::NamedTempFile;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::assert_arrays_eq;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex_cuda::arrow::ArrowArray;
use vortex_ffi::vx_error_free;
use vortex_ffi::vx_error_message;

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
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<ByteBuffer> {
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
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<VortexFile> {
    session
        .open_options()
        .open_buffer(file_bytes(session, array, strategy)?)
}

fn flat_ids_file(session: &VortexSession, rows: u32) -> VortexResult<VortexFile> {
    let ids = PrimitiveArray::from_iter(0..rows).into_array();
    let rows = ids.len();
    let input =
        StructArray::try_new(["ids"].into(), vec![ids], rows, Validity::NonNullable)?.into_array();
    open_file(session, input, Arc::new(FlatLayoutStrategy::default()))
}

#[test]
fn test_projected_scan_zero_batch_rows_preserves_large_layout_span() -> VortexResult<()> {
    let session = session();
    let file = flat_ids_file(&session, 1_000_000)?;
    for columns in [names(&[])?, names(&["ids"])?] {
        let splits = projected_scan(&file, columns, 0)?.full_file_splits()?;
        assert_eq!(splits, [0, 1_000_000]);
    }
    Ok(())
}

#[test]
fn test_projected_scan_exact_batch_rows_with_final_tail() -> VortexResult<()> {
    let session = session();
    let file = flat_ids_file(&session, 1_000)?;
    let expected = StructArray::try_new(
        ["ids"].into(),
        vec![PrimitiveArray::from_iter(0u32..1_000).into_array()],
        1_000,
        Validity::NonNullable,
    )?
    .into_array();
    for columns in [names(&[])?, names(&["ids"])?] {
        let batches: Vec<ArrayRef> = ffi_runtime().block_on(
            projected_scan(&file, columns, 300)?
                .into_array_stream()?
                .try_collect(),
        )?;
        let lengths: Vec<_> = batches.iter().map(|batch| batch.len()).collect();
        assert_eq!(lengths, [300, 300, 300, 100]);
        let actual = ChunkedArray::try_new(batches, expected.dtype().clone())?.into_array();
        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    }
    Ok(())
}

#[test]
fn test_projected_scan_exact_batch_rows_crosses_layout_blocks() -> VortexResult<()> {
    let session = session();
    let input = table()?;
    let columns = names(&["値.x", "ids"])?;
    let expected = input.project(columns.as_ref())?.into_array();
    let input = input.into_array();
    let chunks = ChunkedArray::try_new(
        vec![input.slice(0..2)?, input.slice(2..4)?, input.slice(4..5)?],
        input.dtype().clone(),
    )?
    .into_array();
    let file = open_file(
        &session,
        chunks,
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
    )?;
    assert_eq!(
        projected_scan(&file, columns.clone(), 0)?.full_file_splits()?,
        [0, 2, 4, 5]
    );
    let batches: Vec<ArrayRef> = ffi_runtime().block_on(
        projected_scan(&file, columns, 3)?
            .into_array_stream()?
            .try_collect(),
    )?;
    let lengths: Vec<_> = batches.iter().map(|batch| batch.len()).collect();
    assert_eq!(lengths, [3, 2]);
    let actual = ChunkedArray::try_new(batches, expected.dtype().clone())?.into_array();
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

#[test]
fn test_projected_scan_rejects_unknown_field() -> VortexResult<()> {
    let session = session();
    let file = flat_ids_file(&session, 5)?;
    assert_error(
        projected_scan(&file, names(&["missing"])?, 0),
        "must be a subset of child fields",
    );
    Ok(())
}

#[test]
fn test_projected_scan_rejects_nonstruct_projection() -> VortexResult<()> {
    let session = session();
    let file = open_file(
        &session,
        PrimitiveArray::from_iter(0u32..5).into_array(),
        Arc::new(FlatLayoutStrategy::default()),
    )?;
    assert_error(
        projected_scan(&file, names(&["ids"])?, 0),
        "Select child must return a struct dtype",
    );
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
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy = Arc::new(TableStrategy::new(Arc::clone(&flat), flat));
    let file = open_file(&session, input.into_array(), strategy)?;
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
    // SAFETY: This call owns the returned error and frees it exactly once.
    let message = unsafe { take_error_message(error) }.expect("missing FFI error");
    assert_eq!(status, VX_CUDA_ERR, "{message}");
    assert!(
        message.contains("null CUDA scan columns with nonzero count"),
        "{message}"
    );
    assert_eq!(stream.device_type, -1);
    assert!(stream.release.is_none());
    error = ptr::null_mut();
    // SAFETY: Null output is rejected before any other input is used; error is writable.
    assert_eq!(
        unsafe {
            vx_cuda_scan_path_arrow_device_stream_projected(
                ptr::null(),
                view(""),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null_mut(),
                &raw mut error,
            )
        },
        VX_CUDA_ERR
    );
    // SAFETY: This call owns the returned error and frees it exactly once.
    let message = unsafe { take_error_message(error) }.expect("missing FFI error");
    assert!(
        message.contains("null ArrowDeviceArrayStream output"),
        "{message}"
    );
}

/// # Safety
/// `error` must be null or an owned FFI error, consumed by this call.
unsafe fn take_error_message(error: *mut vx_error) -> Option<String> {
    if error.is_null() {
        return None;
    }
    // SAFETY: The error remains live while its message is copied, then is freed exactly once.
    let message = unsafe { vx_error_message(error).as_str() }
        .map(str::to_owned)
        .unwrap_or_else(|error| error.to_string());
    unsafe { vx_error_free(error) };
    Some(message)
}

struct OwnedDeviceStream(ArrowDeviceArrayStream);

impl Drop for OwnedDeviceStream {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: This owner holds the live stream and releases it exactly once.
            unsafe { release(&raw mut self.0) };
        }
    }
}

fn open_stream(
    session: &VortexSession,
    path: &str,
    options: &vx_cuda_scan_options,
    columns: &[&str],
) -> OwnedDeviceStream {
    let mut output = MaybeUninit::<ArrowDeviceArrayStream>::uninit();
    let mut error = ptr::null_mut();
    let handle = test_session(session.clone());
    let columns: Vec<_> = columns.iter().map(|name| view(name)).collect();
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
    // SAFETY: This call owns both the session handle and any returned error.
    let message = unsafe {
        free_test_session(handle);
        take_error_message(error)
    };
    assert_eq!(
        status,
        VX_CUDA_OK,
        "{}",
        message.as_deref().unwrap_or("no FFI error")
    );
    // SAFETY: A successful call initialized the stream, which owns its session state.
    let stream = OwnedDeviceStream(unsafe { output.assume_init() });
    assert!(message.is_none(), "unexpected FFI error: {message:?}");
    assert!(stream.0.release.is_some(), "missing release");
    stream
}

/// # Safety
/// `array` must be a live Arrow primitive array of `T` on the current CUDA context.
/// Its producer's sync event must have completed before this call.
unsafe fn read_primitive<T: NativePType>(
    array: &ArrowArray,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    assert!(array.release.is_some());
    assert!(array.dictionary.is_null());
    assert_eq!(array.n_buffers, 2);
    assert_eq!(array.n_children, 0);
    assert!(!array.buffers.is_null());
    let len = usize::try_from(array.length)?;
    let offset = usize::try_from(array.offset)?;
    // SAFETY: The live primitive array owns two buffer pointers.
    let buffers = unsafe { std::slice::from_raw_parts(array.buffers, 2) };
    let mut values = vec![T::default(); offset + len];
    // SAFETY: The synchronized data buffer contains offset + len values of T and remains live.
    unsafe { result::memcpy_dtoh_sync(&mut values, buffers[1] as u64) }
        .map_err(|error| vortex_err!("copying Arrow values: {error}"))?;

    let validity = if buffers[0].is_null() {
        assert_eq!(array.null_count, 0);
        match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        }
    } else {
        let mut bytes = vec![0u8; (offset + len).div_ceil(8)];
        // SAFETY: The synchronized bitmap covers offset + len bits and remains live.
        unsafe { result::memcpy_dtoh_sync(&mut bytes, buffers[0] as u64) }
            .map_err(|error| vortex_err!("copying Arrow validity: {error}"))?;
        let bits = BitBuffer::new_with_offset(ByteBuffer::from(bytes), len, offset);
        assert_eq!(array.null_count, i64::try_from(len - bits.true_count())?);
        match nullability {
            Nullability::NonNullable => {
                assert_eq!(array.null_count, 0);
                Validity::NonNullable
            }
            Nullability::Nullable => Validity::from(bits),
        }
    };
    Ok(PrimitiveArray::new(Buffer::from(values).slice(offset..), validity).into_array())
}

/// Read the fixture's projected Int64/UInt32 columns through the public Arrow Device ABI.
///
/// # Safety
/// `array` must be a live batch from the fixture's projected stream, with its schema checked.
unsafe fn read_projected_batch(array: &ArrowDeviceArray) -> VortexResult<ArrayRef> {
    assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
    let context = CudaContext::new(usize::try_from(array.device_id)?)
        .map_err(|error| vortex_err!("opening Arrow device context: {error}"))?;
    context
        .bind_to_thread()
        .map_err(|error| vortex_err!("binding Arrow device context: {error}"))?;
    if !array.sync_event.is_null() {
        // SAFETY: Arrow's CUDA sync_event points to a live event handle owned by this batch.
        unsafe { result::event::synchronize(*array.sync_event.cast::<CUevent>()) }
            .map_err(|error| vortex_err!("waiting for Arrow device batch: {error}"))?;
    }
    let array = &array.array;
    assert!(array.dictionary.is_null());
    assert_eq!(array.offset, 0);
    assert_eq!(array.null_count, 0);
    assert_eq!(array.n_children, 2);
    assert!(!array.children.is_null());
    // SAFETY: The live struct owns two children matching the previously checked schema.
    let fields = unsafe {
        let values = (*array.children).as_ref().expect("missing values child");
        let ids = (*array.children.add(1))
            .as_ref()
            .expect("missing ids child");
        assert_eq!(values.length, array.length);
        assert_eq!(ids.length, array.length);
        vec![
            read_primitive::<i64>(values, Nullability::Nullable)?,
            read_primitive::<u32>(ids, Nullability::NonNullable)?,
        ]
    };
    Ok(StructArray::try_new(
        ["値.x", "ids"].into(),
        fields,
        usize::try_from(array.length)?,
        Validity::NonNullable,
    )?
    .into_array())
}

fn read_projected_batches(stream: &mut ArrowDeviceArrayStream) -> VortexResult<Vec<ArrayRef>> {
    let get_next = stream.get_next.expect("missing get_next");
    let mut batches = Vec::new();
    loop {
        let mut array = ArrowDeviceArray::empty();
        // SAFETY: This live stream owns the callback; array is writable.
        let status = unsafe { get_next(stream, &raw mut array) };
        vortex_ensure!(status == 0, "get_next failed: {}", stream_error(stream));
        if array.array.release.is_none() {
            break;
        }
        // SAFETY: The fixture's schema was checked, and this batch remains live during readback.
        let batch = unsafe { read_projected_batch(&array) };
        release_device_array(&mut array);
        batches.push(batch?);
    }
    Ok(batches)
}

fn check_projected_file(
    input: StructArray,
    block_rows: usize,
    batch_rows: usize,
    expected_lengths: &[usize],
) -> VortexResult<()> {
    let session = session().with_some(CudaSession::try_default()?);
    register_cuda_layout(&session);
    let columns = ["値.x", "ids"];
    let expected = input.project(names(&columns)?.as_ref())?.into_array();
    let mut file = NamedTempFile::new()?;
    file.write_all(&file_bytes(
        &session,
        input.into_array(),
        cuda_write_strategy(&session, block_rows),
    )?)?;
    let path = file
        .path()
        .to_str()
        .ok_or_else(|| vortex_err!("non-UTF-8 test path"))?;
    let options = vx_cuda_scan_options {
        batch_rows,
        ..Default::default()
    };
    let mut stream = open_stream(&session, path, &options, &columns);
    // The stream must retain its session state after the caller releases its session.
    drop(session);
    let schema = stream_schema(&mut stream.0);
    let expected_fields = vec![
        Field::new("値.x", DataType::Int64, true),
        Field::new("ids", DataType::UInt32, false),
    ];
    assert_eq!(Schema::try_from(&schema)?, Schema::new(expected_fields));
    let batches = read_projected_batches(&mut stream.0)?;
    let lengths: Vec<_> = batches.iter().map(|batch| batch.len()).collect();
    assert_eq!(lengths, expected_lengths);
    let actual = ChunkedArray::try_new(batches, expected.dtype().clone())?.into_array();
    assert_arrays_eq!(
        actual,
        expected,
        &mut VortexSession::default().create_execution_ctx()
    );
    Ok(())
}

#[cuda_test]
fn test_projection_gpu_subdivides_large_blocks() -> VortexResult<()> {
    check_projected_file(table()?, 5, 2, &[2, 2, 1])
}

#[cuda_test]
fn test_projection_gpu_preserves_small_block_boundaries() -> VortexResult<()> {
    check_projected_file(table()?, 2, 0, &[2, 2, 1])
}

#[cuda_test]
fn test_projection_gpu_exact_batch_rows_with_final_tail() -> VortexResult<()> {
    let input = StructArray::try_new(
        ["ids", "値.x"].into(),
        vec![
            PrimitiveArray::from_iter(0u32..1_000).into_array(),
            PrimitiveArray::from_option_iter(
                (0i64..1_000).map(|value| (value % 2 == 0).then_some(value)),
            )
            .into_array(),
        ],
        1_000,
        Validity::NonNullable,
    )?;
    check_projected_file(input, 1_000, 300, &[300, 300, 300, 100])
}
