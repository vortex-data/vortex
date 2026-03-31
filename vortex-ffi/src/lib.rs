// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod binary;
mod data_source;
mod dtype;
mod error;
mod expression;
mod file;
mod log;
mod macros;
mod ptype;
mod scan;
mod session;
mod sink;
mod string;
mod struct_array;
mod struct_fields;

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::Arc;
use std::sync::LazyLock;

pub use log::vx_log_level;
use vortex::dtype::FieldName;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::runtime::current::CurrentThreadRuntime;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A shared runtime for all FFI operations.
// TODO(ngates): also create a CurrentThreadPool to manage background worker threads.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: usize) -> Vec<String> {
    #[expect(clippy::expect_used)]
    (0..len)
        .map(|i: usize| unsafe {
            to_string(*ptr.offset(i.try_into().expect("pointer offset overflow")))
        })
        .collect()
}

/// SAFETY: name must be a non-NULL pointer
pub(crate) unsafe fn to_field_name(name: *const c_char) -> VortexResult<FieldName> {
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|e| vortex_err!("{e}"))?;
    let name: Arc<str> = Arc::from(name);
    Ok(name.into())
}

/// SAFETY: names must be a non-NULL pointer valid for reads up to len.
pub(crate) unsafe fn to_field_names(
    names: *const *const c_char,
    len: usize,
) -> VortexResult<Vec<FieldName>> {
    (0..len)
        .map(|i| unsafe {
            let name = *names.offset(i.try_into().vortex_expect("pointer offset overflow"));
            to_field_name(name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;
    use std::sync::Arc;

    use rand::Rng;
    use tempfile::NamedTempFile;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::validity::Validity;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::error::vx_error;
    use crate::error::vx_error_free;
    use crate::error::vx_error_get_message;
    use crate::session::vx_session;
    use crate::sink::vx_array_sink_close;
    use crate::sink::vx_array_sink_open_file;
    use crate::sink::vx_array_sink_push;
    use crate::string::vx_string;

    /// Panic if error is NULL. Free the error if it's not
    pub(crate) fn assert_error(error: *mut vx_error) {
        assert!(!error.is_null(), "Expected error");
        unsafe { vx_error_free(error) };
    }

    /// Panic if error is not NULL.
    pub(crate) fn assert_no_error(error: *mut vx_error) {
        if !error.is_null() {
            let message;
            unsafe {
                message = vx_string::as_str(vx_error_get_message(error)).to_owned();
                vx_error_free(error);
            }
            panic!("{message}");
        }
    }

    fn random_str(length: usize) -> String {
        const CHARSET: &[u8] = b"0123456789";
        let mut rng = rand::thread_rng();

        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    pub const SAMPLE_ROWS: usize = 200;

    /// Write 200 rows of Struct { age=i32, height=i32, name=String } into a
    /// temporary file
    pub(crate) unsafe fn write_sample(session: *const vx_session) -> (NamedTempFile, StructArray) {
        let age = (0..SAMPLE_ROWS as u64).map(Some);
        let age = PrimitiveArray::from_option_iter(age);

        let height = (0..SAMPLE_ROWS as u64).map(|x| Some(200 * x));
        let height = PrimitiveArray::from_option_iter(height);

        let name = (0..SAMPLE_ROWS).map(random_str);
        let name = VarBinViewArray::from_iter_str(name);

        let struct_array = StructArray::try_new(
            ["age", "height", "name"].into(),
            vec![age.into_array(), height.into_array(), name.into_array()],
            SAMPLE_ROWS,
            Validity::NonNullable,
        )
        .unwrap();

        let file = NamedTempFile::new().unwrap();
        let path = CString::new(file.path().to_str().unwrap()).unwrap();
        let dtype = struct_array.dtype();

        unsafe {
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype.clone()));
            let mut error = ptr::null_mut();
            let sink =
                vx_array_sink_open_file(session, path.as_ptr(), vx_dtype_ptr, &raw mut error);
            let array = vx_array::new(Arc::new(struct_array.clone().into_array()));
            vx_array_sink_push(sink, array, &raw mut error);
            vx_array_sink_close(sink, &raw mut error);
            vx_array_free(array);
            vx_dtype_free(vx_dtype_ptr);
        }

        (file, struct_array)
    }
}
