use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrow::IntoArrowArray;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_file::VortexOpenOptions;
use vortex_scalar::Scalar;

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    #[derive(Debug, Clone)]
    enum PType {
        U8 = 0,
        U16 = 1,
        U32 = 2,
        U64 = 3,
        I8 = 4,
        I16 = 5,
        I32 = 6,
        I64 = 7,
        F16 = 8,
        F32 = 9,
        F64 = 10,
    }

    #[derive(Debug, Clone)]
    enum DType {
        Null = 0,
        Bool = 1,
        Primitive = 2,
        Utf8 = 3,
        Binary = 4,
        Struct = 5,
        List = 6,
        Extension = 7,
        Decimal = 8,
    }

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
        type VortexArray;
        type VortexDType;
        type VortexScalar;
        type VortexFile;

        // Array operations
        fn create_dummy() -> Box<VortexArray>;
        fn array_len(array: &VortexArray) -> usize;
        fn array_dtype(array: &VortexArray) -> Box<VortexDType>;
        fn array_is_null(array: &VortexArray, index: usize) -> Result<bool>;
        fn array_scalar_at(array: &VortexArray, index: usize) -> Result<Box<VortexScalar>>;
        fn array_slice(array: &VortexArray, start: usize, stop: usize) -> Result<Box<VortexArray>>;

        // Arrow conversion - returns C ABI structures directly
        fn array_to_arrow(array: &VortexArray) -> Result<CArrowArray>;
        fn array_to_arrow_with_schema(array: &VortexArray) -> Result<ArrowCStructs>;

        // File operations
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn file_row_count(file: &VortexFile) -> u64;
        fn file_read_all(file: &VortexFile) -> Result<Box<VortexArray>>;

        // DType operations
        fn dtype_variant(dtype: &VortexDType) -> DType;
        fn dtype_is_nullable(dtype: &VortexDType) -> bool;
        fn ptype_variant(ptype: &VortexDType) -> PType;

        // Scalar operations
        fn scalar_is_null(scalar: &VortexScalar) -> bool;
        fn scalar_as_bool(scalar: &VortexScalar) -> Result<bool>;
        fn scalar_as_u8(scalar: &VortexScalar) -> Result<u8>;
        fn scalar_as_u16(scalar: &VortexScalar) -> Result<u16>;
        fn scalar_as_u32(scalar: &VortexScalar) -> Result<u32>;
        fn scalar_as_u64(scalar: &VortexScalar) -> Result<u64>;
        fn scalar_as_i8(scalar: &VortexScalar) -> Result<i8>;
        fn scalar_as_i16(scalar: &VortexScalar) -> Result<i16>;
        fn scalar_as_i32(scalar: &VortexScalar) -> Result<i32>;
        fn scalar_as_i64(scalar: &VortexScalar) -> Result<i64>;
        fn scalar_as_f32(scalar: &VortexScalar) -> Result<f32>;
        fn scalar_as_f64(scalar: &VortexScalar) -> Result<f64>;
        fn scalar_as_string(scalar: &VortexScalar) -> Result<String>;
    }
}

pub struct VortexArray {
    inner: ArrayRef,
}

pub struct VortexDType {
    inner: Arc<DType>,
}

pub struct VortexScalar {
    inner: Scalar,
}

pub struct VortexFile {
    inner: vortex_file::VortexFile,
}

// TODO: Remove this once we have a real array constructor
fn create_dummy() -> Box<VortexArray> {
    // Return a dummy array for now - in practice this would be constructed from data
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
    Box::new(VortexArray {
        inner: array.into_array(),
    })
}

fn array_len(array: &VortexArray) -> usize {
    array.inner.len()
}

fn array_dtype(array: &VortexArray) -> Box<VortexDType> {
    Box::new(VortexDType {
        inner: Arc::new(array.inner.dtype().clone()),
    })
}

fn array_is_null(
    array: &VortexArray,
    index: usize,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let scalar = array.inner.scalar_at(index)?;
    Ok(scalar.is_null())
}

fn array_scalar_at(
    array: &VortexArray,
    index: usize,
) -> Result<Box<VortexScalar>, Box<dyn std::error::Error + Send + Sync>> {
    let scalar = array.inner.scalar_at(index)?;
    Ok(Box::new(VortexScalar { inner: scalar }))
}

fn array_slice(
    array: &VortexArray,
    start: usize,
    stop: usize,
) -> Result<Box<VortexArray>, Box<dyn std::error::Error + Send + Sync>> {
    let sliced = array.inner.slice(start, stop)?;
    Ok(Box::new(VortexArray { inner: sliced }))
}

fn dtype_variant(dtype: &VortexDType) -> ffi::DType {
    use vortex_dtype::DType as VDType;

    match dtype.inner.as_ref() {
        VDType::Null => ffi::DType::Null,
        VDType::Bool(_) => ffi::DType::Bool,
        VDType::Primitive(..) => ffi::DType::Primitive,
        VDType::Utf8(_) => ffi::DType::Utf8,
        VDType::Binary(_) => ffi::DType::Binary,
        VDType::Struct(..) => ffi::DType::Struct,
        VDType::List(..) => ffi::DType::List,
        VDType::Extension(_) => ffi::DType::Extension,
        VDType::Decimal(..) => ffi::DType::Primitive,
    }
}

fn dtype_is_nullable(dtype: &VortexDType) -> bool {
    dtype.inner.is_nullable()
}

impl From<vortex_dtype::PType> for ffi::PType {
    fn from(ptype: vortex_dtype::PType) -> Self {
        match ptype {
            vortex_dtype::PType::U8 => ffi::PType::U8,
            vortex_dtype::PType::U16 => ffi::PType::U16,
            vortex_dtype::PType::U32 => ffi::PType::U32,
            vortex_dtype::PType::U64 => ffi::PType::U64,
            vortex_dtype::PType::I8 => ffi::PType::I8,
            vortex_dtype::PType::I16 => ffi::PType::I16,
            vortex_dtype::PType::I32 => ffi::PType::I32,
            vortex_dtype::PType::I64 => ffi::PType::I64,
            vortex_dtype::PType::F16 => ffi::PType::F16,
            vortex_dtype::PType::F32 => ffi::PType::F32,
            vortex_dtype::PType::F64 => ffi::PType::F64,
        }
    }
}

fn ptype_variant(ptype: &VortexDType) -> ffi::PType {
    ptype.inner.as_ptype().into()
}

fn scalar_is_null(scalar: &VortexScalar) -> bool {
    scalar.inner.is_null()
}

fn scalar_as_bool(scalar: &VortexScalar) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let bool_scalar = scalar.inner.as_bool();
    match bool_scalar.value() {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_u8(scalar: &VortexScalar) -> Result<u8, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::U8))?;
    let prim = cast.as_primitive().as_::<u8>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_u16(scalar: &VortexScalar) -> Result<u16, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::U16))?;
    let prim = cast.as_primitive().as_::<u16>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_u32(scalar: &VortexScalar) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::U32))?;
    let prim = cast.as_primitive().as_::<u32>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_u64(scalar: &VortexScalar) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::U64))?;
    let prim = cast.as_primitive().as_::<u64>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_i8(scalar: &VortexScalar) -> Result<i8, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::I8))?;
    let prim = cast.as_primitive().as_::<i8>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_i16(scalar: &VortexScalar) -> Result<i16, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::I16))?;
    let prim = cast.as_primitive().as_::<i16>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_i32(scalar: &VortexScalar) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::I32))?;
    let prim = cast.as_primitive().as_::<i32>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_i64(scalar: &VortexScalar) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::I64))?;
    let prim = cast.as_primitive().as_::<i64>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_f32(scalar: &VortexScalar) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::F32))?;
    let prim = cast.as_primitive().as_::<f32>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_f64(scalar: &VortexScalar) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let cast = scalar.inner.cast(&DType::from(vortex_dtype::PType::F64))?;
    let prim = cast.as_primitive().as_::<f64>()?;
    match prim {
        Some(v) => Ok(v),
        None => Err("Scalar is null".into()),
    }
}

fn scalar_as_string(
    scalar: &VortexScalar,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let utf8 = scalar.inner.as_utf8();
    match utf8.value() {
        Some(s) => Ok(s.to_string()),
        None => Err("Scalar is null".into()),
    }
}

// Arrow conversion functions - export to C ABI
fn array_to_arrow(
    array: &VortexArray,
) -> Result<ffi::CArrowArray, Box<dyn std::error::Error + Send + Sync>> {
    let arrow_array = array.inner.clone().into_arrow_preferred()?;

    // Export to C ABI structures using modern API
    let ffi_array = FFI_ArrowArray::new(&arrow_array.to_data());

    // Directly convert FFI_ArrowArray to CArrowArray using transmute
    // # Safety: The FFI_ArrowArray should have the same memory layout as the C ABI ArrowArray
    let c_array = unsafe {
        // The FFI_ArrowArray should have the same memory layout as the C ABI ArrowArray
        // We can directly transmute between the types
        std::mem::transmute::<FFI_ArrowArray, ffi::CArrowArray>(ffi_array)
    };

    Ok(c_array)
}

fn array_to_arrow_with_schema(
    array: &VortexArray,
) -> Result<ffi::ArrowCStructs, Box<dyn std::error::Error + Send + Sync>> {
    let arrow_array = array.inner.clone().into_arrow_preferred()?;

    // Export to C ABI structures using modern API
    let ffi_array = FFI_ArrowArray::new(&arrow_array.to_data());
    let ffi_schema = FFI_ArrowSchema::try_from(arrow_array.data_type())?;

    // Directly convert FFI_ArrowArray to CArrowArray using transmute
    // # Safety: The FFI_ArrowArray should have the same memory layout as the C ABI ArrowArray
    let c_array = unsafe {
        // The FFI_ArrowArray should have the same memory layout as the C ABI ArrowArray
        // We can directly transmute between the types
        std::mem::transmute::<FFI_ArrowArray, ffi::CArrowArray>(ffi_array)
    };

    // # Safety: The FFI_ArrowSchema should have the same memory layout as the C ABI ArrowSchema
    let c_schema = unsafe { std::mem::transmute::<FFI_ArrowSchema, ffi::CArrowSchema>(ffi_schema) };

    Ok(ffi::ArrowCStructs {
        array: c_array,
        schema: c_schema,
    })
}

// File operations - using blocking operations for simplicity
fn open_file(path: &str) -> Result<Box<VortexFile>, Box<dyn std::error::Error + Send + Sync>> {
    let file = VortexOpenOptions::file().open_blocking(std::path::Path::new(path))?;
    Ok(Box::new(VortexFile { inner: file }))
}

fn file_row_count(file: &VortexFile) -> u64 {
    file.inner.row_count()
}

fn file_read_all(
    file: &VortexFile,
) -> Result<Box<VortexArray>, Box<dyn std::error::Error + Send + Sync>> {
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

    Ok(Box::new(VortexArray { inner: array }))
}
