use std::ffi::{CStr, c_char};

use crate::cpp;

pub struct String<'a> {
    cstring: &'a CStr,
}

impl<'a> String<'a> {
    pub fn from_ptr(ptr: *const c_char) -> Self {
        String {
            cstring: unsafe { CStr::from_ptr(ptr) },
        }
    }
}

impl<'a> Drop for String<'a> {
    fn drop(&mut self) {
        unsafe { cpp::duckdb_free(self.cstring.as_ptr().cast_mut().cast()) };
    }
}
