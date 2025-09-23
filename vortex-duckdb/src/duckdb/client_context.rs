// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::vortex_panic;

use crate::duckdb::ObjectCacheRef;
use crate::{cpp, wrapper};

wrapper!(
    /// A DuckDB client context wrapper.
    ClientContext,
    cpp::duckdb_vx_client_context,
    |_| {}
);

impl ClientContext {
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
}
