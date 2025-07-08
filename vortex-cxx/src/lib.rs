// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::OnceLock;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use tokio::runtime::Runtime;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::arrow::record_batch_reader::VortexRecordBatchReader;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexExpect;
use vortex_file::VortexOpenOptions;

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

    extern "Rust" {
        type VortexFile;

        // File operations
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_scan_to_arrow(file: &VortexFile) -> Result<ArrowCStructs>;
        fn file_scan_to_stream(file: &VortexFile) -> Result<CArrowArrayStream>;
    }
}

pub struct VortexFile {
    inner: vortex_file::VortexFile,
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
