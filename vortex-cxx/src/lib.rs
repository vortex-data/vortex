// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{LazyLock, OnceLock};

use arrow_array::RecordBatchReader;
use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use futures::executor::ThreadPool;
use tokio::runtime::Runtime;
use vortex::arrow::{FromArrowArray, IntoArrowArray, VortexRecordBatchReader};
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect};
use vortex::file::scan::ScanBuilder;
use vortex::file::{VortexOpenOptions, VortexWriteOptions as WriteOptions};
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::stream::ArrayStream;
use vortex::{ArrayRef, IntoArray};

/// The tokio runtime for the write-side.
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::from)
        .vortex_expect("Failed to create tokio runtime")
});

/// The thread pool for the read-side.
static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Thread pool configuration for the read-side.
#[derive(Clone)]
struct ThreadPoolConfig {
    worker_threads: Option<usize>,
}

impl ThreadPoolConfig {
    const fn new() -> Self {
        Self {
            worker_threads: None,
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Get or initialize the thread pool with the default settings
fn get_thread_pool() -> &'static ThreadPool {
    THREAD_POOL.get_or_init(|| {
        create_thread_pool_with_config(&ThreadPoolConfig::default())
            .vortex_expect("Cannot start thread pool")
    })
}

/// Create a thread pool with the given configuration
fn create_thread_pool_with_config(config: &ThreadPoolConfig) -> Result<ThreadPool, std::io::Error> {
    let mut builder = ThreadPool::builder();

    if let Some(worker_threads) = config.worker_threads {
        builder.pool_size(worker_threads);
    }

    builder.create()
}

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    extern "Rust" {
        type VortexFile;
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_scan_builder(file: &VortexFile) -> Result<Box<VortexScanBuilder>>;

        type VortexScanBuilder;
        fn scan_builder_with_row_range(
            builder: &mut VortexScanBuilder,
            row_range_start: u64,
            row_range_end: u64,
        ) -> Result<()>;
        fn scan_builder_with_limit(builder: &mut VortexScanBuilder, limit: usize);
        unsafe fn scan_builder_into_arrow(
            builder: Box<VortexScanBuilder>,
            out_array: *mut u8,
            out_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;

        type VortexWriteOptions;
        fn write_options_new() -> Box<VortexWriteOptions>;
        unsafe fn write_array_stream(
            options: Box<VortexWriteOptions>,
            input_stream: *mut u8,
            path: &str,
        ) -> Result<()>;

        fn configure_thread_pool(worker_threads: usize) -> Result<()>;
    }
}

pub struct VortexFile {
    inner: vortex::file::VortexFile,
}

/// File operations - using blocking operations for simplicity
/// TODO(xinyu): object store (see vortex-ffi)
fn open_file(path: &str) -> Result<Box<VortexFile>, Box<dyn std::error::Error + Send + Sync>> {
    let file = VortexOpenOptions::file().open_blocking(std::path::Path::new(path))?;
    Ok(Box::new(VortexFile { inner: file }))
}

fn file_row_count(file: &VortexFile) -> u64 {
    file.inner.row_count()
}

fn file_scan_builder(
    file: &VortexFile,
) -> Result<Box<VortexScanBuilder>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(VortexScanBuilder {
        inner: file.inner.scan()?,
    }))
}

pub struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
}

fn scan_builder_with_row_range(
    builder: &mut VortexScanBuilder,
    row_range_start: u64,
    row_range_end: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    take_mut::take(&mut builder.inner, |inner| {
        inner.with_row_range(row_range_start..row_range_end)
    });
    Ok(())
}

fn scan_builder_with_limit(builder: &mut VortexScanBuilder, limit: usize) {
    // Overwrite inner without dropping it.
    take_mut::take(&mut builder.inner, |inner| inner.with_limit(limit));
}

/// Arrow Rust FFI interplay with C++ best practices refer to:
/// https://github.com/dora-rs/dora/blob/42775e3612b34d3998b1f7feb7e5df7ec3f8a7bd/examples/c%2B%2B-arrow-dataflow/node-rust-api/main.cc#L12-L19
/// https://github.com/dora-rs/dora/blob/42775e3612b34d3998b1f7feb7e5df7ec3f8a7bd/apis/c%2B%2B/node/src/lib.rs#L211-L212
///
/// # Safety
///
/// out_array should be properly aligned according to C ABI and valid for write.
/// out_schema should be properly aligned according to C ABI and valid for write.
unsafe fn scan_builder_into_arrow(
    builder: Box<VortexScanBuilder>,
    out_array: *mut u8,
    out_schema: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dtype = builder.inner.dtype()?;
    let iter = ArrayIteratorAdapter::new(
        dtype.clone(),
        builder
            .inner
            .into_thread_pool_iter(get_thread_pool().clone())?,
    );
    let reader = VortexRecordBatchReader::try_new(iter)?;

    let arrays: Vec<_> = reader
        .map(|batch| batch.map(|b| ArrayRef::from_arrow(b, false)))
        .collect::<Result<Vec<_>, _>>()?;

    let array = vortex::arrays::ChunkedArray::try_new(arrays, dtype)?
        .into_array()
        .into_arrow_preferred()?;

    let ffi_array = FFI_ArrowArray::new(&array.to_data());
    let ffi_schema = FFI_ArrowSchema::try_from(array.data_type())?;
    // Two discarded attempts recorded here for future reference:
    // 1. Require unsafe transmute and an intermediate CArrowArrayStream defined in cxx shared types
    // Ok(unsafe { std::mem::transmute::<FFI_ArrowArrayStream, ffi::CArrowArrayStream>(stream) })
    // 2. Require Box::new to heap allocate. Also requires an extern "Rust" type `ArrowArrayStream`.
    // Ok(Box::new(ArrowArrayStream { inner: stream }))
    let out_array = out_array as *mut FFI_ArrowArray;
    let out_schema = out_schema as *mut FFI_ArrowSchema;
    // # Safety
    // Arrow C ABI
    unsafe { std::ptr::write(out_array, ffi_array) };
    unsafe { std::ptr::write(out_schema, ffi_schema) };
    Ok(())
}

/// # Safety
///
/// out_stream should be properly aligned according to C ABI and valid for write.
unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dtype = builder.inner.dtype()?;
    let iter = ArrayIteratorAdapter::new(
        dtype,
        builder
            .inner
            .into_thread_pool_iter(get_thread_pool().clone())?,
    );
    let reader = VortexRecordBatchReader::try_new(iter)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C ABI
    unsafe { std::ptr::write(out_stream, stream) };
    Ok(())
}

pub struct VortexWriteOptions {
    inner: WriteOptions,
}

fn write_options_new() -> Box<VortexWriteOptions> {
    Box::new(VortexWriteOptions {
        inner: WriteOptions::default(),
    })
}

/// Convert an ArrowArrayStreamReader to a Vortex ArrayStream
fn arrow_stream_to_vortex_stream(
    reader: ArrowArrayStreamReader,
) -> Result<impl ArrayStream, Box<dyn std::error::Error + Send + Sync>> {
    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|result| {
            result
                .map(|record_batch| ArrayRef::from_arrow(record_batch, false))
                .map_err(VortexError::from)
        }),
    );

    Ok(array_iter.into_array_stream())
}

/// # Safety
///
/// input_stream should be valid FFI_ArrowArrayStream.
/// See [`FFI_ArrowArrayStream::from_raw`]
unsafe fn write_array_stream(
    options: Box<VortexWriteOptions>,
    input_stream: *mut u8,
    path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = path.to_string();

    let stream_reader =
        unsafe { ArrowArrayStreamReader::from_raw(input_stream as *mut FFI_ArrowArrayStream) }?;

    let vortex_stream = arrow_stream_to_vortex_stream(stream_reader)?;

    RUNTIME.block_on(async {
        let file = tokio::fs::File::create(path).await?;

        options.inner.write(file, vortex_stream).await?;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
}

/// Configure the read-side thread pool with the specified number of worker threads
///
/// If the thread pool has already been initialized, this function will return an error.
fn configure_thread_pool(
    worker_threads: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if thread pool has already been initialized
    if THREAD_POOL.get().is_some() {
        return Err("Thread pool has already been initialized. ".into());
    }

    THREAD_POOL.get_or_init(|| {
        create_thread_pool_with_config(&ThreadPoolConfig {
            worker_threads: Some(worker_threads),
        })
        .vortex_expect("Cannot start thread pool")
    });

    Ok(())
}

#[cfg(feature = "gen_test_data")]
mod gen_test_data {
    #[cxx::bridge(namespace = "vortex::ffi")]
    mod ffi {
        extern "Rust" {
            fn generate_test_vortex_file(output_path: &str) -> Result<()>;
        }
    }
    fn generate_test_vortex_file(
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            use tokio::fs::File;
            use vortex::IntoArray;
            use vortex::arrays::{PrimitiveArray, StructArray};
            use vortex::buffer::buffer;
            use vortex::dtype::FieldNames;
            use vortex::error::VortexExpect;
            use vortex::file::VortexWriteOptions;
            use vortex::validity::Validity;
            // Create a simple test array
            let test_array = PrimitiveArray::new(
                buffer![10i32, 20i32, 30i32, 40i32, 50i32],
                Validity::NonNullable,
            )
            .into_array();

            let struct_array = StructArray::try_new(
                FieldNames::from_iter(vec!["a".to_string(), "b".to_string()]),
                vec![test_array.clone(), test_array.clone()],
                5,
                Validity::NonNullable,
            )
            .vortex_expect("Failed to create test array");

            let mut file = File::create(&output_path)
                .await
                .vortex_expect("Failed to create test file");

            VortexWriteOptions::default()
                .write(&mut file, struct_array.to_array_stream())
                .await
                .vortex_expect("Failed to write test data to file");
        });
        Ok(())
    }
}
