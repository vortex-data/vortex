use std::sync::OnceLock;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use tokio::runtime::Runtime;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::arrow::record_batch_reader::VortexRecordBatchReader;
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
        buffers: usize,      // pointer as usize
        children: usize,     // pointer as usize
        dictionary: usize,   // pointer as usize
        release: usize,      // function pointer as usize
        private_data: usize, // pointer as usize
    }

    struct CArrowSchema {
        format: usize,   // pointer to c_char as usize
        name: usize,     // pointer to c_char as usize
        metadata: usize, // pointer to c_char as usize
        flags: i64,
        n_children: i64,
        children: usize,     // pointer as usize
        dictionary: usize,   // pointer as usize
        release: usize,      // function pointer as usize
        private_data: usize, // pointer as usize
    }

    struct CArrowArrayStream {
        get_schema: usize, // function pointer: int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out)
        get_next: usize, // function pointer: int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out)
        get_last_error: usize, // function pointer: const char* (*get_last_error)(struct ArrowArrayStream*)
        release: usize,        // function pointer: void (*release)(struct ArrowArrayStream*)
        private_data: usize,   // pointer: void* private_data
    }

    extern "Rust" {
        type VortexFile;
        // type VortexArrayStream;

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
        use futures::stream::StreamExt;
        let mut arrays = Vec::new();
        let mut stream = std::pin::pin!(stream);
        while let Some(array) = stream.next().await {
            arrays.push(array?);
        }

        // If we have multiple arrays, we need to concatenate them
        if arrays.is_empty() {
            Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "No data in file",
            ))
        } else if arrays.len() == 1 {
            Ok(arrays.into_iter().next().unwrap())
        } else {
            Ok(ChunkedArray::from_iter(arrays).into_array())
        }
    })?;

    // Convert to Arrow and then to C ABI
    let arrow_array = array.into_arrow_preferred()?;
    let ffi_array = FFI_ArrowArray::new(&arrow_array.to_data());
    let ffi_schema = FFI_ArrowSchema::try_from(arrow_array.data_type())?;

    // Convert to our C-compatible structs
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
    Ok(unsafe { std::mem::transmute::<FFI_ArrowArrayStream, ffi::CArrowArrayStream>(stream) })
}
