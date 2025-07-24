// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{Array, RecordBatchReader};
use arrow_schema::{Schema, SchemaRef};
use arrow_select::concat::concat_batches;
use vortex::ArrayRef;
use vortex::arrow::VortexRecordBatchReader;
use vortex::file::VortexOpenOptions;
use vortex::file::scan::ScanBuilder;
use vortex::iter::ArrayIteratorAdapter;

use crate::get_thread_pool;

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
        unsafe fn scan_builder_with_output_schema(
            builder: &mut VortexScanBuilder,
            output_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_arrow(
            builder: Box<VortexScanBuilder>,
            out_array: *mut u8,
            out_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
    }
}

struct VortexFile {
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

struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
    output_schema: Option<SchemaRef>,
}

fn file_scan_builder(
    file: &VortexFile,
) -> Result<Box<VortexScanBuilder>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(VortexScanBuilder {
        inner: file.inner.scan()?,
        output_schema: None,
    }))
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

unsafe fn scan_builder_with_output_schema(
    builder: &mut VortexScanBuilder,
    output_schema: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(output_schema as *mut FFI_ArrowSchema) };
    builder.output_schema = Some(Arc::new(Schema::try_from(&ffi_schema)?));
    Ok(())
}

/// Convert a VortexScanBuilder into a VortexRecordBatchReader
fn scan_builder_to_reader(
    builder: Box<VortexScanBuilder>,
) -> Result<impl RecordBatchReader + 'static, Box<dyn std::error::Error + Send + Sync>> {
    let dtype = builder.inner.dtype()?;
    let iter = ArrayIteratorAdapter::new(
        dtype,
        builder
            .inner
            .into_thread_pool_iter(get_thread_pool().clone())?,
    );
    let reader = if let Some(schema) = builder.output_schema {
        VortexRecordBatchReader::try_new_with_schema(iter, schema)?
    } else {
        VortexRecordBatchReader::try_new(iter)?
    };
    Ok(reader)
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
    let reader = scan_builder_to_reader(builder)?;

    let schema = reader.schema();
    let batches: Result<Vec<_>, _> = reader.into_iter().collect();
    let batches = batches?;
    let combined = concat_batches(&schema, &batches)?;

    let struct_array: arrow_array::StructArray = combined.into();

    let ffi_array = FFI_ArrowArray::new(&struct_array.to_data());
    let ffi_schema = FFI_ArrowSchema::try_from(struct_array.data_type())?;
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
    let reader = scan_builder_to_reader(builder)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C ABI
    unsafe { std::ptr::write(out_stream, stream) };
    Ok(())
}
