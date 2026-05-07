// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::ClientContextRef;
use crate::duckdb::drop_boxed;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A borrowed DuckDB object-cache entry handle.
    ObjectCacheEntry,
    cpp::duckdb_vx_object_cache_entry,
    cpp::duckdb_vx_object_cache_entry_free
);

impl ObjectCacheEntryRef {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        unsafe { cpp::duckdb_vx_object_cache_entry_get_data(self.as_ptr()) }
    }
}

impl ClientContextRef {
    /// Retrieve a cloned Rust value from DuckDB's per-database object cache.
    ///
    /// `object_type` must match the type string used when storing the value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every object stored under the `(key,
    /// object_type)` pair was inserted as the same Rust type `T`.
    pub unsafe fn object_cache_get_cloned<T: Clone>(
        &self,
        key: &str,
        object_type: &CStr,
    ) -> VortexResult<Option<T>> {
        let key = cache_key(key)?;
        let entry = unsafe {
            cpp::duckdb_vx_object_cache_get(
                self.as_ptr(),
                key.as_ptr(),
                key.as_bytes().len(),
                object_type.as_ptr(),
            )
        };
        if entry.is_null() {
            return Ok(None);
        }

        let entry = unsafe { ObjectCacheEntry::own(entry) };
        let data = entry.data_ptr();
        if data.is_null() {
            return Ok(None);
        }

        Ok(Some(unsafe { (&*data.cast::<T>()).clone() }))
    }

    /// Store a Rust value in DuckDB's per-database object cache.
    ///
    /// `estimated_memory` is reported to DuckDB's object cache in bytes for
    /// eviction accounting.
    pub fn object_cache_put<T: Send + Sync + 'static>(
        &self,
        key: &str,
        object_type: &CStr,
        estimated_memory: usize,
        value: T,
    ) -> VortexResult<()> {
        let key = cache_key(key)?;
        let estimated_memory = cpp::idx_t::try_from(estimated_memory)
            .map_err(|_| vortex_err!("object cache memory estimate does not fit idx_t"))?;
        let data = Box::into_raw(Box::new(value));
        let state = unsafe {
            cpp::duckdb_vx_object_cache_put(
                self.as_ptr(),
                key.as_ptr(),
                key.as_bytes().len(),
                object_type.as_ptr(),
                estimated_memory,
                data.cast(),
                Some(drop_boxed::<T>),
            )
        };
        if state != cpp::duckdb_state::DuckDBSuccess {
            vortex_bail!("failed to store object in DuckDB object cache");
        }
        Ok(())
    }
}

fn cache_key(key: &str) -> VortexResult<CString> {
    CString::new(key).map_err(|_| vortex_err!("object cache key contains an interior NUL byte"))
}

#[cfg(test)]
mod tests {
    use vortex::error::VortexResult;

    use crate::duckdb::Database;

    #[test]
    fn object_cache_round_trip_clones_stored_value() -> VortexResult<()> {
        let db = Database::open_in_memory()?;
        let conn = db.connect()?;
        let ctx = conn.client_context()?;

        assert_eq!(
            unsafe { ctx.object_cache_get_cloned::<String>("cache-key", c"vortex_test") }?,
            None
        );

        ctx.object_cache_put("cache-key", c"vortex_test", 5, String::from("value"))?;

        assert_eq!(
            unsafe { ctx.object_cache_get_cloned::<String>("cache-key", c"vortex_test") }?,
            Some(String::from("value"))
        );
        assert_eq!(
            unsafe { ctx.object_cache_get_cloned::<String>("cache-key", c"other_type") }?,
            None
        );

        Ok(())
    }
}
