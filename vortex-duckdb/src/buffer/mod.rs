use std::os::raw::c_void;

use duckdb::ffi::{duckdb_vector_buffer, duckdb_wrap_external_vector_buffer, external_buffer};
use vortex_buffer::ByteBuffer;

#[derive(Clone)]
#[repr(C)]
pub struct FFIDuckDBBufferInternal {
    pub inner: Box<ByteBuffer>,
}

// This CANNOT be copied or cloned since the void* inner is actually an Arc, and the ref counter
// will be incorrect.
#[repr(C)]
pub struct ExternalBuffer {
    pub inner: *mut c_void,
}

impl From<FFIDuckDBBufferInternal> for ExternalBuffer {
    fn from(buffer: FFIDuckDBBufferInternal) -> Self {
        let ptr = Box::into_raw(buffer.inner) as *mut c_void;
        ExternalBuffer { inner: ptr }
    }
}

impl From<ExternalBuffer> for FFIDuckDBBufferInternal {
    fn from(buffer: ExternalBuffer) -> Self {
        let inner: Box<ByteBuffer> = unsafe { Box::from_raw(buffer.inner.cast()) };
        FFIDuckDBBufferInternal { inner }
    }
}

// This will free a single FFIDuckDBBuffer, however due to cloning there might be more
// references to the underlying ByteBuffer that will not be freed in this call.
#[unsafe(no_mangle)]
unsafe extern "C" fn ExternalBuffer_free(buffer: external_buffer) {
    let internal: Box<FFIDuckDBBufferInternal> = unsafe { Box::from_raw(buffer.cast()) };
    drop(internal)
}

pub unsafe fn new_cpp_vector_buffer(buffer: *mut ExternalBuffer) -> duckdb_vector_buffer {
    unsafe { duckdb_wrap_external_vector_buffer(buffer.cast(), Some(ExternalBuffer_free)) }
}

#[cfg(test)]
mod tests {

    use vortex_buffer::ByteBuffer;

    use crate::buffer::{ExternalBuffer, FFIDuckDBBufferInternal};

    #[test]
    fn test_buff_drop() {
        let buffer = FFIDuckDBBufferInternal {
            inner: Box::new(ByteBuffer::from(vec![1, 2, 3])),
        };

        assert!(buffer.inner.inner().is_unique());

        let buffer_er: ExternalBuffer = buffer.clone().into();
        let buffer_back: FFIDuckDBBufferInternal = buffer_er.into();

        assert!(!buffer_back.inner.inner().is_unique());
        assert!(!buffer.inner.inner().is_unique());
    }
}
