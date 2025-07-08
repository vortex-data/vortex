use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_schema::{Schema, SchemaRef};
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_file::VortexOpenOptions;

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

    extern "Rust" {
        type VortexFile;
        // type VortexArrayStream;

        // File operations
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_scan_to_arrow(file: &VortexFile) -> Result<ArrowCStructs>;
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
    let rt = tokio::runtime::Runtime::new()?;

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

// // Stream implementation
// pub struct VortexArrayStream {
//     arrays: Vec<arrow_array::ArrayRef>,
//     schema: SchemaRef,
//     current_index: usize,
//     last_error: Option<CString>,
// }

// impl VortexArrayStream {
//     fn new(arrays: Vec<arrow_array::ArrayRef>, schema: SchemaRef) -> Self {
//         Self {
//             arrays,
//             schema,
//             current_index: 0,
//             last_error: None,
//         }
//     }
// }

// unsafe fn stream_get_schema(stream: &VortexArrayStream, out: *mut ffi::CArrowSchema) -> i32 {
//     if out.is_null() {
//         return -1;
//     }

//     match FFI_ArrowSchema::try_from(stream.schema.as_ref()) {
//         Ok(ffi_schema) => {
//             let c_schema = std::mem::transmute::<FFI_ArrowSchema, ffi::CArrowSchema>(ffi_schema);
//             *out = c_schema;
//             0
//         }
//         Err(_) => -1,
//     }
// }

// unsafe fn stream_get_next(stream: &VortexArrayStream, out: *mut ffi::CArrowArray) -> i32 {
//     if out.is_null() {
//         return -1;
//     }

//     // Since we can't mutate stream through &, we'll need to track index differently
//     // For now, let's return the first array and then end the stream
//     if stream.current_index >= stream.arrays.len() {
//         // End of stream - set array to released state
//         (*out).release = 0; // Use 0 to indicate released
//         return 0;
//     }

//     let array = &stream.arrays[stream.current_index];
//     let ffi_array = FFI_ArrowArray::new(&array.to_data());
//     let c_array = std::mem::transmute::<FFI_ArrowArray, ffi::CArrowArray>(ffi_array);
//     *out = c_array;

//     0
// }

// fn stream_get_last_error(stream: &VortexArrayStream) -> *const c_char {
//     match &stream.last_error {
//         Some(err) => err.as_ptr(),
//         None => ptr::null(),
//     }
// }

// fn file_scan_to_stream(
//     file: &VortexFile,
// ) -> Result<Box<VortexArrayStream>, Box<dyn std::error::Error + Send + Sync>> {
//     // Create a runtime for async operations
//     let rt = tokio::runtime::Runtime::new()?;

//     let arrays = rt.block_on(async {
//         let stream = file.inner.scan()?.into_array_stream()?;
//         use futures::stream::StreamExt;
//         let mut arrays = Vec::new();
//         let mut stream = std::pin::pin!(stream);
//         while let Some(array) = stream.next().await {
//             arrays.push(array?);
//         }
//         Ok::<_, Box<dyn std::error::Error + Send + Sync>>(arrays)
//     })?;

//     if arrays.is_empty() {
//         return Err("No data in file".into());
//     }

//     // Convert to Arrow arrays
//     let arrow_arrays: Result<Vec<_>, _> = arrays
//         .into_iter()
//         .map(|array| array.into_arrow_preferred())
//         .collect();
//     let arrow_arrays = arrow_arrays?;

//     // Get schema from first array
//     let schema = Arc::new(Schema::new(vec![arrow_schema::Field::new(
//         "data",
//         arrow_arrays[0].data_type().clone(),
//         true,
//     )]));

//     // Create stream data
//     let stream_data = Box::new(VortexArrayStream::new(arrow_arrays, schema));
//     Ok(stream_data)
// }
