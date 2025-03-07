use std::ffi::{CStr, c_char};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn duckdb_hello() -> *const c_char {
    CStr::from_bytes_with_nul(b"Hello, world! (from rust)\0")
        .unwrap()
        .as_ptr()
}
