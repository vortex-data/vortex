// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod binary;
mod dtype;
mod error;
mod expression;
mod file;
mod log;
mod macros;
mod ptype;
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
