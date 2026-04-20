// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use futures::stream::TryStreamExt;
use vortex::array::ArrayRef;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::SESSION;
use crate::expr::Expr;

pub(crate) struct VortexFile {
    inner: vortex::file::VortexFile,
}

impl VortexFile {
    pub(crate) fn row_count(&self) -> u64 {
        self.inner.row_count()
    }

    pub(crate) fn scan_builder(&self) -> Result<Box<VortexScanBuilder>> {
        Ok(Box::new(VortexScanBuilder {
            inner: self.inner.scan()?,
            output_schema: None,
        }))
    }
}

/// File operations - using blocking operations for simplicity
/// TODO(xinyu): object store (see vortex-ffi)
pub(crate) fn open_file(path: &str) -> Result<Box<VortexFile>> {
    let file = RUNTIME.block_on(SESSION.open_options().open_path(std::path::Path::new(path)))?;
    Ok(Box::new(VortexFile { inner: file }))
}

pub(crate) fn open_file_from_buffer(data: &[u8]) -> Result<Box<VortexFile>> {
    let buffer = Buffer::from(data.to_vec());
    let file = SESSION.open_options().open_buffer(buffer)?;
    Ok(Box::new(VortexFile { inner: file }))
}

pub(crate) struct VortexScanBuilder {
    inner: ScanBuilder<ArrayRef>,
    output_schema: Option<SchemaRef>,
}

impl VortexScanBuilder {
    pub(crate) fn with_filter(&mut self, filter: Box<Expr>) {
        take_mut::take(&mut self.inner, |inner| inner.with_filter(filter.inner));
    }

    pub(crate) fn with_filter_ref(&mut self, filter: &Expr) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_filter(filter.inner.clone())
        });
    }

    pub(crate) fn with_projection(&mut self, filter: Box<Expr>) {
        take_mut::take(&mut self.inner, |inner| inner.with_projection(filter.inner));
    }

    pub(crate) fn with_projection_ref(&mut self, filter: &Expr) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_projection(filter.inner.clone())
        });
    }

    pub(crate) fn with_row_range(&mut self, row_range_start: u64, row_range_end: u64) {
        take_mut::take(&mut self.inner, |inner| {
            inner.with_row_range(row_range_start..row_range_end)
        });
    }

    pub(crate) fn with_include_by_index(&mut self, include_by_index: &[u64]) {
        let selection = Selection::IncludeByIndex(Buffer::copy_from(include_by_index));
        take_mut::take(&mut self.inner, |inner| inner.with_selection(selection));
    }

    pub(crate) fn with_limit(&mut self, limit: usize) {
        take_mut::take(&mut self.inner, |inner| inner.with_limit(limit as u64));
    }

    pub(crate) unsafe fn with_output_schema(&mut self, output_schema: *mut u8) -> Result<()> {
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(output_schema as *mut FFI_ArrowSchema) };
        self.output_schema = Some(Arc::new(Schema::try_from(&ffi_schema)?));
        Ok(())
    }
}

/// # Safety
///
/// out_stream should be properly aligned according to the Arrow C stream interface and valid for write.
pub(crate) unsafe fn scan_builder_into_stream(
    builder: Box<VortexScanBuilder>,
    out_stream: *mut u8,
) -> Result<()> {
    let schema = match builder.output_schema {
        Some(schema) => schema,
        None => {
            let dtype = builder.inner.dtype()?;
            let arrow_schema = dtype.to_arrow_schema()?;
            Arc::new(arrow_schema)
        }
    };
    let reader = builder.inner.into_record_batch_reader(schema, &*RUNTIME)?;
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

pub(crate) struct ThreadsafeCloneableReader {
    inner: Box<dyn ThreadsafeCloneableReaderTrait>,
}

pub(crate) fn scan_builder_into_threadsafe_cloneable_reader(
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
    let data_type = DataType::Struct(schema.fields().clone());

    let stream = builder
        .inner
        .map(move |b| {
            b.execute_arrow(Some(&data_type), &mut LEGACY_SESSION.create_execution_ctx())
                .map(|struct_array| RecordBatch::from(struct_array.as_struct()))
        })
        .into_stream()?
        .map_err(|e| ArrowError::ExternalError(Box::new(e)));

    let iter = RUNTIME.block_on_stream_thread_safe(|_h| stream);
    let rbr = RecordBatchIteratorAdapter::new(iter, schema);

    Ok(Box::new(ThreadsafeCloneableReader {
        inner: Box::new(rbr),
    }))
}

impl ThreadsafeCloneableReader {
    pub(crate) fn clone_a_stream(&self, out_stream: *mut u8) {
        let cloned_reader = self.inner.clone_boxed();
        let stream = FFI_ArrowArrayStream::new(cloned_reader);
        let out_stream = out_stream as *mut FFI_ArrowArrayStream;
        // # Safety
        // Arrow C stream interface
        unsafe { std::ptr::write(out_stream, stream) };
    }
}
