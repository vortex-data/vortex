// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::io::Write;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::TryStreamExt;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStream;
use vortex::arrow::ArrowSessionExt;
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

#[test]
fn test_projection_names_are_owned_and_zero_count_means_all() -> VortexResult<()> {
    let parsed = {
        let name = String::from("値.x");
        names(&[&name, ""])?
    };
    assert_eq!(parsed, ["値.x", ""]);
    // SAFETY: Zero count ignores the pointer, including a null pointer.
    assert!(unsafe { scan_columns(ptr::null(), 0) }?.is_empty());
    assert!(names(&[])?.is_empty());
    let empty = vx_view {
        ptr: ptr::null(),
        len: 0,
    };
    // SAFETY: A null, zero-length view is a valid empty name.
    assert_eq!(unsafe { scan_columns(&raw const empty, 1) }?, [""]);
    Ok(())
}

#[test]
fn test_projection_rejects_invalid_names_and_counts() -> VortexResult<()> {
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
        let error = unsafe { scan_columns(columns, count) }
            .err()
            .ok_or_else(|| vortex_err!("expected invalid projection"))?;
        assert!(error.to_string().contains(message), "{error}");
    }
    Ok(())
}

fn session() -> VortexSession {
    VortexSession::default().with_handle(ffi_runtime().handle())
}

fn table(rows: usize) -> VortexResult<ArrayRef> {
    Ok(StructArray::try_new(
        ["ids", "unused", "値.x"].into(),
        vec![
            PrimitiveArray::from_iter((0u32..5).take(rows)).into_array(),
            PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0].into_iter().take(rows))
                .into_array(),
            PrimitiveArray::from_option_iter(
                [Some(10i64), None, Some(30), None, Some(50)]
                    .into_iter()
                    .take(rows),
            )
            .into_array(),
        ],
        rows,
        Validity::NonNullable,
    )?
    .into_array())
}

fn file_bytes(session: &VortexSession, array: ArrayRef, cuda: bool) -> VortexResult<ByteBuffer> {
    let strategy: Arc<dyn LayoutStrategy> = if cuda {
        register_cuda_layout(session);
        WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().only_cuda_compatible())
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
            .build()
    } else {
        let flat = Arc::new(FlatLayoutStrategy::default());
        Arc::new(TableStrategy::new(
            Arc::<FlatLayoutStrategy>::clone(&flat),
            flat,
        ))
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

#[test]
fn test_projection_cpu_scan_order_dtype_empty_and_defaults() -> VortexResult<()> {
    let session = session();
    for rows in [0, 5] {
        let input = table(rows)?;
        let file =
            session
                .open_options()
                .open_buffer(file_bytes(&session, input.clone(), false)?)?;
        for columns in [vec!["値.x", "ids"], vec!["ids"], vec![]] {
            let expected = if columns.is_empty() {
                input.clone()
            } else {
                input
                    .clone()
                    .execute::<StructArray>(&mut session.create_execution_ctx())?
                    .project(names(&columns)?.as_ref())?
                    .into_array()
            };
            for batch_rows in [0, 2] {
                let scan = projected_scan(&file, names(&columns)?, batch_rows)?;
                assert_eq!(scan.dtype()?, *expected.dtype());
                let stream = scan.into_array_stream()?;
                assert_eq!(stream.dtype(), expected.dtype());
                let actual = ffi_runtime().block_on(stream.read_all())?;
                let mut ctx = session.create_execution_ctx();
                assert_eq!(
                    session
                        .arrow()
                        .execute_arrow(actual, None, &mut ctx)?
                        .to_data(),
                    session
                        .arrow()
                        .execute_arrow(expected.clone(), None, &mut ctx)?
                        .to_data(),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_projection_cpu_rejects_unknown_names_and_non_struct() -> VortexResult<()> {
    let session = session();
    for rows in [0, 5] {
        let file =
            session
                .open_options()
                .open_buffer(file_bytes(&session, table(rows)?, false)?)?;
        for name in ["missing", "IDS", "値", "値.x.child"] {
            let error = projected_scan(&file, names(&[name])?, 0)
                .err()
                .ok_or_else(|| vortex_err!("expected unknown column error"))?;
            assert!(
                error.to_string().contains("unknown CUDA scan column"),
                "{error}"
            );
        }
    }
    let input = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let file = session
        .open_options()
        .open_buffer(file_bytes(&session, input.clone(), false)?)?;
    let error = projected_scan(&file, names(&["ids"])?, 0)
        .err()
        .ok_or_else(|| vortex_err!("expected non-struct projection error"))?;
    assert!(
        error.to_string().contains("requires a struct file dtype"),
        "{error}"
    );
    let actual = ffi_runtime().block_on(
        projected_scan(&file, names(&[])?, 0)?
            .into_array_stream()?
            .read_all(),
    )?;
    let mut ctx = session.create_execution_ctx();
    assert_eq!(
        session
            .arrow()
            .execute_arrow(actual, None, &mut ctx)?
            .to_data(),
        session
            .arrow()
            .execute_arrow(input, None, &mut ctx)?
            .to_data(),
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
    let file = session
        .open_options()
        .open_buffer(file_bytes(&session, table(5)?, false)?)?;
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
    let batches: Vec<_> = ffi_runtime().block_on(
        projected_scan(&file, names(&["値.x", "ids"])?, 2)?
            .into_array_stream()?
            .try_collect(),
    )?;
    assert_eq!(
        batches.iter().map(|batch| batch.len()).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(source.rejected.load(Ordering::Relaxed), 0);
    // The same reader must fail without projection, proving the guard actually observes reads.
    let error = ffi_runtime()
        .block_on(
            projected_scan(&file, names(&[])?, 0)?
                .into_array_stream()?
                .read_all(),
        )
        .err()
        .ok_or_else(|| vortex_err!("full scan must request unused data"))?;
    assert!(
        error
            .to_string()
            .contains("unselected column segment requested"),
        "{error}"
    );
    assert!(source.rejected.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn test_projection_ffi_validation_without_cuda() -> VortexResult<()> {
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
    Ok(())
}

struct LocalFile(PathBuf);

impl LocalFile {
    fn new(bytes: &[u8]) -> VortexResult<Self> {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vortex-cuda-ffi-projection-{}-{}.vortex",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let result = Self(path);
        file.write_all(bytes)?;
        Ok(result)
    }
}

impl Drop for LocalFile {
    fn drop(&mut self) {
        drop(std::fs::remove_file(&self.0));
    }
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

#[cuda_test]
fn test_projection_gpu_local_file_schema_batches_empty_and_defaults() -> VortexResult<()> {
    let session = session().with_some(CudaSession::try_default()?);
    for rows in [0, 5] {
        let file = LocalFile::new(&file_bytes(&session, table(rows)?, true)?)?;
        let path = file
            .0
            .to_str()
            .ok_or_else(|| vortex_err!("non-UTF-8 test path"))?;
        for projected in [true, false] {
            for options in [
                None,
                Some(vx_cuda_scan_options {
                    flags: VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES,
                    batch_rows: 2,
                }),
            ] {
                let options = options.as_ref().map_or(ptr::null(), ptr::from_ref);
                let mut output = MaybeUninit::<ArrowDeviceArrayStream>::uninit();
                let mut error = ptr::null_mut();
                let handle = test_session(session.clone());
                let columns = [view("値.x"), view("ids")];
                // SAFETY: All borrowed inputs and writable outputs are live for this call.
                let status = unsafe {
                    if projected {
                        vx_cuda_scan_path_arrow_device_stream_projected(
                            handle,
                            view(path),
                            options,
                            columns.as_ptr(),
                            columns.len(),
                            output.as_mut_ptr(),
                            &raw mut error,
                        )
                    } else {
                        vx_cuda_scan_path_arrow_device_stream_with_options(
                            handle,
                            view(path),
                            options,
                            output.as_mut_ptr(),
                            &raw mut error,
                        )
                    }
                };
                unsafe { free_test_session(handle) };
                assert_eq!(status, VX_CUDA_OK);
                assert!(error.is_null());
                // SAFETY: A successful call initialized the stream, which owns its session state.
                let mut stream = unsafe { output.assume_init() };
                let mut schema = FFI_ArrowSchema::empty();
                let get_schema = stream
                    .get_schema
                    .ok_or_else(|| vortex_err!("missing get_schema"))?;
                // SAFETY: This live stream owns the callback; schema is writable.
                assert_eq!(
                    unsafe { get_schema(&raw mut stream, (&raw mut schema).cast()) },
                    0,
                    "{}",
                    stream_error(&mut stream)
                );
                let expected_fields = if projected {
                    vec![
                        Field::new("値.x", DataType::Int64, true),
                        Field::new("ids", DataType::UInt32, false),
                    ]
                } else {
                    vec![
                        Field::new("ids", DataType::UInt32, false),
                        Field::new("unused", DataType::Float64, false),
                        Field::new("値.x", DataType::Int64, true),
                    ]
                };
                assert_eq!(Schema::try_from(&schema)?, Schema::new(expected_fields));
                let get_next = stream
                    .get_next
                    .ok_or_else(|| vortex_err!("missing get_next"))?;
                let mut lengths = Vec::new();
                loop {
                    let mut array = empty_device_array();
                    // SAFETY: This live stream owns the callback; array is writable.
                    assert_eq!(
                        unsafe { get_next(&raw mut stream, &raw mut array) },
                        0,
                        "{}",
                        stream_error(&mut stream)
                    );
                    if array.array.release.is_none() {
                        break;
                    }
                    assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
                    assert_eq!(array.array.n_children, if projected { 2 } else { 3 });
                    lengths.push(array.array.length);
                    unsafe { release_device_array(&mut array) };
                }
                assert_eq!(lengths.iter().sum::<i64>(), rows as i64);
                if rows > 0 && !options.is_null() {
                    assert_eq!(lengths, [2, 2, 1]);
                }
                let release = stream
                    .release
                    .ok_or_else(|| vortex_err!("missing release"))?;
                // SAFETY: Both objects are live and released exactly once.
                unsafe {
                    release_schema(&mut schema);
                    release(&raw mut stream);
                }
            }
        }
    }
    Ok(())
}
