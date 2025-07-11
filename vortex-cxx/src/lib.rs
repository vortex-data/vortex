// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::OnceLock;

use arrow_array::RecordBatchReader;
use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use prost::Message;
use tokio::runtime::Runtime;
use vortex::arrow::IntoArrowArray;
use vortex::arrow::record_batch_reader::VortexRecordBatchReader;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect};
use vortex::expr::deserialize_expr;
use vortex::file::scan::ScanBuilder;
use vortex::file::{VortexOpenOptions, VortexWriteOptions as WriteOptions};
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::proto::expr::Expr;
use vortex::stream::{
    ArrayStream, ArrayStreamExt, ArrayStreamToIterator, AsyncRuntime, SendableArrayStream,
};
use vortex::{ArrayRef, TryIntoArray};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Runtime configuration for the tokio runtime
#[derive(Clone)]
struct RuntimeConfig {
    worker_threads: Option<usize>,
}

impl RuntimeConfig {
    const fn new() -> Self {
        Self {
            worker_threads: None,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Get or initialize the tokio runtime with the default settings
fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        create_runtime_with_config(&RuntimeConfig::default()).vortex_expect("Cannot start runtime")
    })
}

/// Create a tokio runtime with the given configuration
fn create_runtime_with_config(config: &RuntimeConfig) -> Result<Runtime, std::io::Error> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();

    if let Some(worker_threads) = config.worker_threads {
        builder.worker_threads(worker_threads);
    }

    builder.build()
}

/// Runtime adapter for vortex-cxx that uses the global static runtime
struct CxxRuntimeAdapter;

impl AsyncRuntime for CxxRuntimeAdapter {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        get_runtime().block_on(fut)
    }
}

/// Convenience function to create an ArrayStreamToIterator with the global runtime
fn array_stream_to_iterator<S>(stream: S) -> ArrayStreamToIterator<S, CxxRuntimeAdapter>
where
    S: ArrayStream + Unpin + Send,
{
    ArrayStreamToIterator::new(stream, CxxRuntimeAdapter)
}

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    extern "Rust" {
        type VortexFile;
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_scan_builder(file: &VortexFile) -> Result<Box<VortexScanBuilder>>;

        type VortexScanBuilder;
        fn scan_builder_set_filter(builder: &mut VortexScanBuilder, filter: &[u8]) -> Result<()>;
        fn scan_builder_set_limit(builder: &mut VortexScanBuilder, limit: usize);
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

        fn configure_runtime(worker_threads: usize) -> Result<()>;
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

fn scan_builder_set_filter(
    builder: &mut VortexScanBuilder,
    filter: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = deserialize_expr(&Expr::decode(filter)?)
        .map_err(|e| e.with_context("deserializing filter expr"))?;
    take_mut::take(&mut builder.inner, |inner| inner.with_filter(filter));
    Ok(())
}

fn scan_builder_set_limit(builder: &mut VortexScanBuilder, limit: usize) {
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
    let rt = get_runtime();
    let array = rt
        .block_on(async {
            let stream = builder.inner.into_array_stream()?;
            stream.read_all().await
        })?
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
    std::ptr::write(out_array, ffi_array);
    std::ptr::write(out_schema, ffi_schema);
    Ok(())
}

/// # Safety
///
/// out_stream should be properly aligned according to C ABI and valid for write.
unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let iter =
        array_stream_to_iterator(builder.inner.into_array_stream()?.boxed() as SendableArrayStream);
    let reader = VortexRecordBatchReader::try_new(iter)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C ABI
    std::ptr::write(out_stream, stream);
    Ok(())
}

/// Convert an ArrowArrayStreamReader to a Vortex ArrayStream
fn arrow_stream_to_vortex_stream(
    reader: ArrowArrayStreamReader,
) -> Result<impl ArrayStream, Box<dyn std::error::Error + Send + Sync>> {
    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|result| {
            result
                .map_err(|e| VortexError::from(e))
                .and_then(|record_batch| record_batch.try_into_array())
        }),
    );

    Ok(array_iter.into_array_stream())
}

pub struct VortexWriteOptions {
    inner: WriteOptions,
}

fn write_options_new() -> Box<VortexWriteOptions> {
    Box::new(VortexWriteOptions {
        inner: WriteOptions::default(),
    })
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
    let stream_reader =
        ArrowArrayStreamReader::from_raw(input_stream as *mut FFI_ArrowArrayStream)?;

    let vortex_stream = arrow_stream_to_vortex_stream(stream_reader)?;

    let rt = get_runtime();

    rt.block_on(async {
        let file = tokio::fs::File::create(path).await?;
        options.inner.write(file, vortex_stream).await?;
        Ok(())
    })
}

/// Configure the tokio runtime with the specified number of worker threads
///
/// If the runtime has already been initialized, this function will return an error.
fn configure_runtime(
    worker_threads: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if runtime has already been initialized
    if RUNTIME.get().is_some() {
        return Err("Runtime has already been initialized. ".into());
    }

    RUNTIME.get_or_init(|| {
        create_runtime_with_config(&RuntimeConfig {
            worker_threads: Some(worker_threads),
        })
        .vortex_expect("Cannot start runtime")
    });

    Ok(())
}
