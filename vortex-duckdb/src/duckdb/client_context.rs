// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use crate::cpp;
use crate::duckdb::Value;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A DuckDB client context wrapper.
    ClientContext,
    cpp::duckdb_client_context,
    cpp::duckdb_destroy_client_context
);

// SAFETY: ClientContext carries an opaque pointer. It is safe to send/share across threads
// under the same guarantees: the underlying DuckDB context is valid for the connection
// lifetime and DuckDB synchronizes internal state.
unsafe impl Send for ClientContextRef {}
unsafe impl Sync for ClientContextRef {}

impl ClientContextRef {
    /// Erases the lifetime of this reference, returning a `&'static ClientContextRef`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying `ClientContext` outlives all uses of the
    /// returned reference. In practice, the `ClientContext` is owned by the `Connection`
    /// and lives as long as the connection, so this is safe as long as the connection is kept alive.
    pub unsafe fn erase_lifetime(&self) -> &'static Self {
        unsafe { &*(self as *const Self) }
    }

    /// Try to get the current value of a configuration setting.
    /// Returns None if the setting doesn't exist.
    pub fn try_get_current_setting(&self, key: &CStr) -> Option<Value> {
        unsafe {
            let value_ptr =
                cpp::duckdb_client_context_try_get_current_setting(self.as_ptr(), key.as_ptr());
            if value_ptr.is_null() {
                None
            } else {
                Some(Value::own(value_ptr))
            }
        }
    }
}
