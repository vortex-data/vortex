// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use vortex::error::vortex_panic;

use crate::cpp;
use crate::duckdb::ObjectCache;
use crate::duckdb::OwnedValue;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A DuckDB client context wrapper.
    ClientContext,
    cpp::duckdb_client_context,
    |_| {
        // No cleanup is necessary since the client context is owned by the connection and will
        // be valid for the connection's lifetime.
    }
);

// SAFETY: ClientContext carries an opaque pointer. It is safe to send/share across threads
// under the same guarantees: the underlying DuckDB context is valid for the connection
// lifetime and DuckDB synchronizes internal state.
unsafe impl Send for OwnedClientContext {}
unsafe impl Sync for OwnedClientContext {}

impl Clone for OwnedClientContext {
    fn clone(&self) -> Self {
        // ClientContext is a lightweight wrapper around an opaque pointer owned by the connection.
        // Cloning just creates another wrapper around the same pointer.
        // Since the destructor is a no-op, this is safe.
        unsafe { Self::own(self.as_ptr()) }
    }
}

impl ClientContext {
    /// Creates an owned handle from a borrowed reference.
    ///
    /// This is safe because ClientContext has a no-op destructor.
    pub fn to_owned_handle(&self) -> OwnedClientContext {
        unsafe { OwnedClientContext::own(self.as_ptr()) }
    }

    /// Get the object cache for this client context.
    pub fn object_cache(&self) -> &ObjectCache {
        unsafe {
            let cache = cpp::duckdb_client_context_get_object_cache(self.as_ptr());
            if cache.is_null() {
                vortex_panic!("Failed to get object cache from client context");
            }
            ObjectCache::borrow(cache)
        }
    }

    /// Try to get the current value of a configuration setting.
    /// Returns None if the setting doesn't exist.
    pub fn try_get_current_setting(&self, key: &CStr) -> Option<OwnedValue> {
        unsafe {
            let value_ptr =
                cpp::duckdb_client_context_try_get_current_setting(self.as_ptr(), key.as_ptr());
            if value_ptr.is_null() {
                None
            } else {
                Some(OwnedValue::own(value_ptr))
            }
        }
    }
}
