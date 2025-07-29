// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::{Schema, SchemaRef};
use vortex::ArrayRef;
use vortex::buffer::Buffer;
use vortex::file::VortexOpenOptions;
use vortex::scan::ScanBuilder;

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
        fn scan_builder_with_include_by_index(
            builder: &mut VortexScanBuilder,
            include_by_index: &[u64],
        ) -> Result<()>;
        fn scan_builder_with_limit(builder: &mut VortexScanBuilder, limit: usize);
        unsafe fn scan_builder_with_output_schema(
            builder: &mut VortexScanBuilder,
            output_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
        type ThreadsafeCloneableReader;
        fn scan_builder_into_threadsafe_cloneable_reader(
            builder: Box<VortexScanBuilder>,
        ) -> Result<Box<ThreadsafeCloneableReader>>;
        unsafe fn threadsafe_cloneable_reader_clone_a_stream(
            reader: &ThreadsafeCloneableReader,
            out_stream: *mut u8,
        );
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

fn scan_builder_with_include_by_index(
    builder: &mut VortexScanBuilder,
    include_by_index: &[u64],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let selection = vortex::scan::Selection::IncludeByIndex(Buffer::copy_from(include_by_index));
    take_mut::take(&mut builder.inner, |inner| inner.with_selection(selection));
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

/// # Safety
///
/// out_stream should be properly aligned according to the Arrow C stream interface and valid for write.
unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let schema = match builder.output_schema {
        Some(schema) => schema,
        None => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = dtype.to_arrow_schema()?;
            Arc::new(arrow_schema)
        }
    };
    let reader = builder.inner.into_record_batch_reader(schema)?;
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C stream interface
    unsafe { std::ptr::write(out_stream, stream) };
    Ok(())
}

trait ThreadsafeCloneableReaderTrait: RecordBatchReader + Send + 'static {
    fn clone_boxed(&self) -> Box<dyn ThreadsafeCloneableReaderTrait>;
}

impl<T> ThreadsafeCloneableReaderTrait for T
where
    T: RecordBatchReader + Send + Clone + 'static,
{
    fn clone_boxed(&self) -> Box<dyn ThreadsafeCloneableReaderTrait> {
        Box::new(self.clone())
    }
}

struct ThreadsafeCloneableReader {
    inner: Box<dyn ThreadsafeCloneableReaderTrait>,
}

fn scan_builder_into_threadsafe_cloneable_reader(
    builder: Box<VortexScanBuilder>,
) -> Result<Box<ThreadsafeCloneableReader>, Box<dyn std::error::Error + Send + Sync>> {
    let schema = match builder.output_schema {
        Some(schema) => schema,
        None => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = dtype.to_arrow_schema()?;
            Arc::new(arrow_schema)
        }
    };
    let reader = builder.inner.into_record_batch_reader(schema)?;
    Ok(Box::new(ThreadsafeCloneableReader {
        inner: Box::new(reader),
    }))
}

fn threadsafe_cloneable_reader_clone_a_stream(
    reader: &ThreadsafeCloneableReader,
    out_stream: *mut u8,
) {
    let cloned_reader = reader.inner.clone_boxed();
    let stream = FFI_ArrowArrayStream::new(cloned_reader);
    let out_stream = out_stream as *mut FFI_ArrowArrayStream;
    // # Safety
    // Arrow C stream interface
    unsafe { std::ptr::write(out_stream, stream) };
}
