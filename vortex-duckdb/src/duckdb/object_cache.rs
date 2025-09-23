// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::os::raw::c_void;

use vortex::error::{VortexUnwrap, vortex_err};

use crate::{cpp, lifetime_wrapper};

/// Custom deleter function for Box<T> allocated in Rust
unsafe extern "C-unwind" fn rust_box_deleter<T>(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut T);
        }
    }
}

// ObjectCache is a wrapper around a DuckDB object cache.
// We only implement ObjectCacheRef since duckdb only has a single object cache per client,
// context which is never owned.
lifetime_wrapper!(ObjectCache, cpp::duckdb_vx_object_cache, |_| {}, [ref]);

impl ObjectCacheRef<'_> {
    /// Store an entry in the object cache with the given key.
    /// The entry will be converted to an opaque pointer and stored.
    /// Uses a proper deleter to ensure memory is freed when the cache entry is removed.
    pub fn put<T: 'static>(&self, key: &str, entry: T) -> *mut T {
        let key_cstr = CString::new(key)
            .map_err(|e| vortex_err!("invalid key: {}", e))
            .vortex_unwrap();
        let opaque_ptr = Box::into_raw(Box::new(entry));

        unsafe {
            cpp::duckdb_vx_object_cache_put(
                self.as_ptr(),
                key_cstr.as_ptr(),
                opaque_ptr as *mut c_void,
                Some(rust_box_deleter::<T>),
            );
        }
        opaque_ptr
    }

    /// Retrieve an entry from the object cache with the given key.
    /// Returns None if the key is not found.
    pub fn get<T>(&self, key: &str) -> Option<&T> {
        let key_cstr = CString::new(key)
            .map_err(|e| vortex_err!("invalid key: {}", e))
            .vortex_unwrap();

        unsafe {
            let opaque_ptr = cpp::duckdb_vx_object_cache_get(self.as_ptr(), key_cstr.as_ptr());
            (!opaque_ptr.is_null())
                .then_some((opaque_ptr as *const T).as_ref())
                .flatten()
        }
    }
}
// This is Send + Sync since the cache has a mutex wrapper.
unsafe impl Send for ObjectCacheRef<'_> {}
unsafe impl Sync for ObjectCacheRef<'_> {}
