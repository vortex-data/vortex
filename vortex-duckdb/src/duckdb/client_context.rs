// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use vortex::error::vortex_panic;

use crate::cpp;
use crate::duckdb::ObjectCacheRef;
use crate::duckdb::Value;
use crate::wrapper;

wrapper!(
    /// A DuckDB client context wrapper.
    ClientContext,
    cpp::duckdb_vx_client_context,
    |_| {}
);

#[derive(Clone, Copy)]
pub(crate) struct SendableClientContext(cpp::duckdb_vx_client_context);
// SAFETY: SendableClientContext carries the same opaque pointer as ClientContext. It is safe to
// send/share across threads under the same guarantees as ClientContext: the underlying DuckDB
// context is valid for the connection lifetime and DuckDB synchronizes internal state.
unsafe impl Send for SendableClientContext {}
unsafe impl Sync for SendableClientContext {}

impl ClientContext {
    pub(crate) fn as_sendable(&self) -> SendableClientContext {
        SendableClientContext(self.as_ptr())
    }

    /// Get the object cache for this client context.
    pub fn object_cache(&self) -> ObjectCacheRef<'static> {
        unsafe {
            let cache = cpp::duckdb_vx_client_context_get_object_cache(self.as_ptr());
            if cache.is_null() {
                vortex_panic!("Failed to get object cache from client context");
            }
            ObjectCacheRef::borrow(cache)
        }
    }

    /// Try to get the current value of a configuration setting.
    /// Returns None if the setting doesn't exist.
    pub fn try_get_current_setting(&self, key: &CStr) -> Option<Value> {
        unsafe {
            let value_ptr =
                cpp::duckdb_vx_client_context_try_get_current_setting(self.as_ptr(), key.as_ptr());
            if value_ptr.is_null() {
                None
            } else {
                Some(Value::own(value_ptr))
            }
        }
    }
}

impl SendableClientContext {
    pub(crate) fn as_ptr(self) -> cpp::duckdb_vx_client_context {
        self.0
    }
}
