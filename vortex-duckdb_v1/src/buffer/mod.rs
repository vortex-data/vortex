use duckdb::ffi::{duckdb_vector_buffer, duckdb_wrap_external_vector_buffer, external_buffer};
use vortex::buffer::ByteBuffer;

// This will free a single FFIDuckDBBuffer, however due to cloning there might be more
// references to the underlying ByteBuffer that will not be freed in this call.
#[unsafe(no_mangle)]
unsafe extern "C" fn vx_duckdb_byte_buffer_free(buffer: external_buffer) {
    let buffer: Box<ByteBuffer> = unsafe { Box::from_raw(buffer.cast()) };
    drop(buffer)
}

pub fn new_buffer(buffer: ByteBuffer) -> duckdb_vector_buffer {
    unsafe {
        duckdb_wrap_external_vector_buffer(
            Box::into_raw(Box::new(buffer)).cast(),
            Some(vx_duckdb_byte_buffer_free),
        )
    }
}
