// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::os::raw::c_void;

use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;

use crate::cpp;
use crate::lifetime_wrapper;

/// Custom deleter function for Box<T> allocated in Rust
unsafe extern "C-unwind" fn rust_box_deleter<T>(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr as *mut T));
        }
    }
}

lifetime_wrapper!(ObjectCache, cpp::duckdb_vx_object_cache, |_| {
    vortex_panic!("ObjectCache is owned by the DatabaseInstance, not by Rust")
});

impl ObjectCacheRef {
    /// Erases the lifetime of this reference, returning a `&'static ObjectCacheRef`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying `ObjectCache` outlives all uses of the
    /// returned reference. In practice, the `ObjectCache` is owned by the `DatabaseInstance`
    /// and lives as long as the database, so this is safe as long as the database is kept alive.
    pub unsafe fn erase_lifetime(&self) -> &'static Self {
        unsafe { &*(self as *const Self) }
    }

    /// Store an entry in the object cache with the given key.
    /// The entry will be converted to an opaque pointer and stored.
    /// Uses a proper deleter to ensure memory is freed when the cache entry is removed.
    pub fn put<T: 'static>(&self, key: &str, entry: T) -> *mut T {
        let key_cstr = CString::new(key)
            .map_err(|e| vortex_err!("invalid key: {}", e))
            .vortex_expect("object cache key should be valid C string");
        let opaque_ptr = Box::into_raw(Box::new(entry));

        // Pass 0 to allow eviction by DuckDB, u64::MAX to prevent eviction, otherwise provide an
        // estimate of the size of the object in bytes.
        let estimated_size: u64 = 0;

        unsafe {
            cpp::duckdb_vx_object_cache_put(
                self.as_ptr(),
                key_cstr.as_ptr(),
                opaque_ptr as *mut c_void,
                estimated_size,
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
            .vortex_expect("object cache key should be valid C string");

        unsafe {
            let opaque_ptr = cpp::duckdb_vx_object_cache_get(self.as_ptr(), key_cstr.as_ptr());
            (!opaque_ptr.is_null())
                .then_some(opaque_ptr.cast::<T>().as_ref())
                .flatten()
        }
    }
}

// SAFETY: Send + Sync since the cache has a mutex wrapper on the C++ side.
unsafe impl Send for ObjectCacheRef {}
unsafe impl Sync for ObjectCacheRef {}
