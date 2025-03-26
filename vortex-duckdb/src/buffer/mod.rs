use std::os::raw::c_void;

use duckdb::ffi::duckdb_vector;
use vortex_buffer::ByteBuffer;

#[derive(Clone)]
#[repr(C)]
pub struct FFIDuckDBBufferInternal {
    pub inner: Box<ByteBuffer>,
}

// This CANNOT be copied or cloned since the void* inner is actually an Arc, and the ref counter
// will be incorrect.
#[repr(C)]
pub struct FFIDuckDBBuffer {
    pub inner: *mut c_void,
}

impl From<FFIDuckDBBufferInternal> for FFIDuckDBBuffer {
    fn from(buffer: FFIDuckDBBufferInternal) -> Self {
        let ptr = Box::into_raw(buffer.inner) as *mut c_void;
        FFIDuckDBBuffer { inner: ptr }
    }
}

impl From<FFIDuckDBBuffer> for FFIDuckDBBufferInternal {
    fn from(buffer: FFIDuckDBBuffer) -> Self {
        let inner: Box<ByteBuffer> = unsafe { Box::from_raw(buffer.inner.cast()) };
        FFIDuckDBBufferInternal { inner }
    }
}

// This will free a single FFIDuckDBBuffer, however due to cloning there might be more
// references to the underlying ByteBuffer that will not be freed in this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDuckDBBuffer_free(buffer: *mut FFIDuckDBBufferInternal) {
    drop(unsafe { Box::from_raw(buffer) })
}

#[repr(C)]
#[allow(dead_code)]
pub struct CppVectorBuffer {
    pub ptr: *mut c_void,
}

#[allow(dead_code)]
unsafe extern "C" {
    pub fn NewCppVectorBuffer(buffer: *mut FFIDuckDBBuffer) -> *mut CppVectorBuffer;

    pub fn AssignBufferToVec(vector: duckdb_vector, buffer: *mut CppVectorBuffer);
}

#[cfg(test)]
mod tests {

    use vortex_buffer::ByteBuffer;

    use crate::buffer::{FFIDuckDBBuffer, FFIDuckDBBufferInternal};

    #[test]
    fn test_buff_drop() {
        let buffer = FFIDuckDBBufferInternal {
            inner: Box::new(ByteBuffer::from(vec![1, 2, 3])),
        };

        assert!(buffer.inner.inner().is_unique());

        let buffer_er: FFIDuckDBBuffer = buffer.clone().into();
        let buffer_back: FFIDuckDBBufferInternal = buffer_er.into();

        assert!(!buffer_back.inner.inner().is_unique());
        assert!(!buffer.inner.inner().is_unique());
    }
}
