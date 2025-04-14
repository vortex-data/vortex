use std::ffi::{c_char, c_int};

#[repr(C)]
pub struct FFIError {
    pub code: c_int,
    pub message: *const c_char,
}
