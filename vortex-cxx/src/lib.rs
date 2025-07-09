// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::OnceLock;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use prost::Message;
use tokio::runtime::Runtime;
use vortex::ArrayRef;
use vortex::arrow::IntoArrowArray;
use vortex::arrow::record_batch_reader::VortexRecordBatchReader;
use vortex::error::VortexExpect;
use vortex::expr::deserialize_expr;
use vortex::file::VortexOpenOptions;
use vortex::file::scan::ScanBuilder;
use vortex::proto::expr::Expr;
use vortex::stream::ArrayStreamExt;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {

    struct ArrowCStructs {
        array: CArrowArray,
        schema: CArrowSchema,
    }

    // C-compatible structs for Arrow C ABI
    struct CArrowArray {
        length: i64,
        null_count: i64,
        offset: i64,
        n_buffers: i64,
        n_children: i64,
        buffers: usize,
        children: usize,
        dictionary: usize,
        release: usize,
        private_data: usize,
    }

    struct CArrowSchema {
        format: usize,
        name: usize,
        metadata: usize,
        flags: i64,
        n_children: i64,
        children: usize,
        dictionary: usize,
        release: usize,
        private_data: usize,
    }

    struct CArrowArrayStream {
        get_schema: usize,
        get_next: usize,
        get_last_error: usize,
        release: usize,
        private_data: usize,
    }

    // ScanOptions struct that can be passed from C++ to Rust
    struct ScanOptions {
        filter_expr: &'static [u8],
        has_filter: bool,
        limit: usize,
        has_limit: bool,
    }

    extern "Rust" {
        type ArrowArrayStream;

        type VortexFile;
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_scan_builder(file: &VortexFile) -> Result<Box<VortexScanBuilder>>;
        fn file_scan_to_arrow(file: &VortexFile) -> Result<ArrowCStructs>;
        fn file_scan_to_stream(file: &VortexFile) -> Result<CArrowArrayStream>;

        type VortexScanBuilder;
        // TODO: figure out the best practice for passing &[u8] from C++ to Rust
        // fn scan_builder_set_filter(
        //     builder: &mut VortexScanBuilder,
        //     filter: &'static [u8],
        // ) -> Result<()>;
        fn scan_builder_set_limit(builder: &mut VortexScanBuilder, limit: usize);
        unsafe fn scan_builder_to_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
    }
}

/// --- A wrapper around the Vortex file. ---
pub struct VortexFile {
    inner: vortex::file::VortexFile,
}

pub struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
}

pub struct ArrowArrayStream {
    inner: FFI_ArrowArrayStream,
}

fn file_scan_builder(
    file: &VortexFile,
) -> Result<Box<VortexScanBuilder>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(VortexScanBuilder {
        inner: file.inner.scan()?,
    }))
}

fn scan_builder_set_filter(
    builder: &mut VortexScanBuilder,
    filter: &'static [u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = deserialize_expr(&Expr::decode(filter)?)
        .map_err(|e| e.with_context("deserializing filter expr"))?;
    // The implementation of `take_mut` includes a copy of the inner value, but we assume the compiler can optimize it away.
    take_mut::take(&mut builder.inner, |inner| inner.with_filter(filter));
    Ok(())
}

fn scan_builder_set_limit(builder: &mut VortexScanBuilder, limit: usize) {
    take_mut::take(&mut builder.inner, |inner| inner.with_limit(limit));
}

// File operations - using blocking operations for simplicity
fn open_file(path: &str) -> Result<Box<VortexFile>, Box<dyn std::error::Error + Send + Sync>> {
    let file = VortexOpenOptions::file().open_blocking(std::path::Path::new(path))?;
    Ok(Box::new(VortexFile { inner: file }))
}

fn file_row_count(file: &VortexFile) -> u64 {
    file.inner.row_count()
}

fn file_scan_to_arrow(
    file: &VortexFile,
) -> Result<ffi::ArrowCStructs, Box<dyn std::error::Error + Send + Sync>> {
    // Create a runtime for async operations
    let rt = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .vortex_expect("Cannot start runtime")
    });

    let array = rt.block_on(async {
        let stream = file.inner.scan()?.into_array_stream()?;
        stream.read_all().await
    })?;

    // Convert to Arrow and then to C ABI
    let arrow_array = array.into_arrow_preferred()?;
    let ffi_array = FFI_ArrowArray::new(&arrow_array.to_data());
    let ffi_schema = FFI_ArrowSchema::try_from(arrow_array.data_type())?;

    // Convert to our C-compatible structs
    // # Safety
    // Arrow C ABI
    let c_array = unsafe { std::mem::transmute::<FFI_ArrowArray, ffi::CArrowArray>(ffi_array) };
    let c_schema = unsafe { std::mem::transmute::<FFI_ArrowSchema, ffi::CArrowSchema>(ffi_schema) };

    Ok(ffi::ArrowCStructs {
        array: c_array,
        schema: c_schema,
    })
}

fn file_scan_to_stream(
    file: &VortexFile,
) -> Result<ffi::CArrowArrayStream, Box<dyn std::error::Error + Send + Sync>> {
    let iter = file.inner.scan()?.into_array_iter()?;
    let reader = VortexRecordBatchReader::try_new(iter)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    // # Safety
    // Arrow C ABI
    Ok(unsafe { std::mem::transmute::<FFI_ArrowArrayStream, ffi::CArrowArrayStream>(stream) })
}

/// Arrow Rust FFI interplay with C++ best practices refer to:
/// https://github.com/dora-rs/dora/blob/42775e3612b34d3998b1f7feb7e5df7ec3f8a7bd/examples/c%2B%2B-arrow-dataflow/node-rust-api/main.cc#L12-L19
/// https://github.com/dora-rs/dora/blob/42775e3612b34d3998b1f7feb7e5df7ec3f8a7bd/apis/c%2B%2B/node/src/lib.rs#L211-L212
unsafe fn scan_builder_to_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let iter = builder.inner.into_array_iter()?;
    let reader = VortexRecordBatchReader::try_new(iter)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    // Two discarded attempts:
    // 1. Require unsafe transmute and an intermediate CArrowArrayStream defined in cxx shared types
    // Ok(unsafe { std::mem::transmute::<FFI_ArrowArrayStream, ffi::CArrowArrayStream>(stream) })
    // 2. Require Box::new to heap allocate. Also requires an extern Rust type `ArrowArrayStream`.
    // Ok(Box::new(ArrowArrayStream { inner: stream }))
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C ABI
    std::ptr::write(out_stream, stream);
    Ok(())
}
