// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;

use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::DatabaseRef;
use crate::duckdb::Value;
use crate::duckdb_try;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A DuckDB configuration instance.
    Config,
    cpp::duckdb_config,
    cpp::duckdb_destroy_config
);

impl Config {
    /// Creates a new DuckDB configuration.
    pub fn new() -> VortexResult<Self> {
        let mut ptr: cpp::duckdb_config = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_create_config(&raw mut ptr) },
            "Failed to create DuckDB config"
        );

        Ok(unsafe { Self::own(ptr) })
    }
}

impl ConfigRef {
    /// Sets a key-value configuration parameter.
    pub fn set(&mut self, key: &str, value: &str) -> VortexResult<()> {
        let key_cstr =
            CString::new(key).map_err(|_| vortex_err!("Invalid key: contains null bytes"))?;
        let value_cstr =
            CString::new(value).map_err(|_| vortex_err!("Invalid value: contains null bytes"))?;

        duckdb_try!(
            unsafe {
                cpp::duckdb_set_config(self.as_ptr(), key_cstr.as_ptr(), value_cstr.as_ptr())
            },
            "Failed to set config parameter '{}' to '{}'",
            key,
            value
        );

        Ok(())
    }

    /// Gets the value of a configuration parameter that was previously set.
    /// Returns None if the parameter was never set on this Config instance.
    pub fn get(&self, key: &str) -> Option<Value> {
        let key_cstr = CString::new(key).ok()?;

        let mut value: cpp::duckdb_value = ptr::null_mut();
        let result = unsafe {
            cpp::duckdb_vx_get_config_value(self.as_ptr(), key_cstr.as_ptr(), &raw mut value)
        };

        (result == cpp::duckdb_state::DuckDBSuccess && !value.is_null())
            .then(|| unsafe { Value::own(value) })
    }

    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|value| {
            let c_str = unsafe { cpp::duckdb_vx_value_to_string(value.as_ptr()) };

            if !c_str.is_null() {
                let rust_str = unsafe { CStr::from_ptr(c_str).to_string_lossy().into_owned() };

                // Free the C string allocated by our function
                unsafe { cpp::duckdb_free(c_str as *mut c_void) };

                return Some(rust_str);
            }
            None
        })
    }

    /// Checks if a configuration key has been set on this config instance.
    pub fn has_key(&self, key: &str) -> bool {
        let Ok(key_cstr) = CString::new(key) else {
            return false;
        };

        let result = unsafe { cpp::duckdb_vx_config_has_key(self.as_ptr(), key_cstr.as_ptr()) };
        result == 1
    }

    /// Returns the number of configuration parameters available in DuckDB.
    pub fn count() -> usize {
        unsafe { cpp::duckdb_config_count() }
    }

    /// Gets information about a configuration option by index.
    /// Returns (name, description) if the index is valid, None otherwise.
    pub fn get_config_flag(index: usize) -> VortexResult<Option<(String, String)>> {
        let mut name_ptr: *const c_char = ptr::null();
        let mut desc_ptr: *const c_char = ptr::null();

        let result =
            unsafe { cpp::duckdb_get_config_flag(index, &raw mut name_ptr, &raw mut desc_ptr) };

        if result != cpp::duckdb_state::DuckDBSuccess {
            return Ok(None);
        }

        let name = unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() };
        let description = unsafe { CStr::from_ptr(desc_ptr).to_string_lossy().into_owned() };

        Ok(Some((name, description)))
    }

    /// Returns a list of all available configuration options.
    pub fn list_available_options() -> VortexResult<Vec<(String, String)>> {
        let count = Self::count();
        let mut options = Vec::with_capacity(count);

        for i in 0..count {
            if let Some((name, desc)) = Self::get_config_flag(i)? {
                options.push((name, desc));
            }
        }

        Ok(options)
    }

    /// Add a new extension option.
    pub fn add_extension_options(
        &self,
        name: &str,
        description: &str,
        logical_type: LogicalType,
        default_value: Value,
    ) -> VortexResult<()> {
        let name_cstr =
            CString::new(name).map_err(|_| vortex_err!("Invalid name: contains null bytes"))?;
        let desc_cstr = CString::new(description)
            .map_err(|_| vortex_err!("Invalid description: contains null bytes"))?;

        duckdb_try!(unsafe {
            cpp::duckdb_vx_add_extension_option(
                self.as_ptr(),
                name_cstr.as_ptr(),
                desc_cstr.as_ptr(),
                logical_type.as_ptr(),
                default_value.as_ptr(),
            )
        });
        Ok(())
    }
}

use crate::duckdb::LogicalType;

impl DatabaseRef {
    pub fn config(&self) -> &ConfigRef {
        unsafe { Config::borrow(cpp::duckdb_vx_database_get_config(self.as_ptr())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duckdb::Database;

    #[test]
    fn test_config_creation() {
        let config = Config::new();
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_set_and_get() {
        let mut config = Config::new().unwrap();

        // Set some values
        assert!(config.set("memory_limit", "1GB").is_ok());
        assert!(config.set("threads", "2").is_ok());

        // Verify they can be retrieved
        assert_eq!(config.get_str("memory_limit"), Some("1GB".to_string()));
        assert_eq!(config.get_str("threads"), Some("2".to_string()));

        // Non-existent key should return None
        assert_eq!(config.get_str("nonexistent_key"), None);
    }

    #[test]
    fn test_config_get_all() {
        let mut config = Config::new().unwrap();

        assert!(config.set("memory_limit", "512MB").is_ok());
        assert!(config.set("threads", "4").is_ok());
        assert!(config.set("max_memory", "1GB").is_ok());

        assert_eq!(config.get_str("memory_limit"), Some("512MB".to_string()));
        assert_eq!(config.get_str("threads"), Some("4".to_string()));
        assert_eq!(config.get_str("max_memory"), Some("1GB".to_string()));
    }

    #[test]
    fn test_config_update_value() {
        let mut config = Config::new().unwrap();

        // Set initial value
        assert!(config.set("threads", "2").is_ok());
        assert_eq!(config.get_str("threads"), Some("2".to_string()));

        // Update the value
        assert!(config.set("threads", "8").is_ok());
        assert_eq!(config.get_str("threads"), Some("8".to_string()));
    }

    #[test]
    fn test_config_persistence_through_database() {
        // Create config with specific settings
        let mut config = Config::new().unwrap();
        config.set("memory_limit", "256MB").unwrap();
        config.set("threads", "1").unwrap();

        // Verify values are stored before using
        assert_eq!(config.get_str("memory_limit"), Some("256MB".to_string()));
        assert_eq!(config.get_str("threads"), Some("1".to_string()));

        // Use config to create database (this consumes the config)
        let db = Database::open_in_memory_with_config(config);
        assert!(db.is_ok());

        // Verify database was created successfully
        let conn = db.unwrap().connect();
        assert!(conn.is_ok());
    }

    #[test]
    fn test_config_invalid_key() {
        let mut config = Config::new().unwrap();
        let result = config.set("key\0with\0nulls", "value");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_value() {
        let mut config = Config::new().unwrap();
        let result = config.set("key", "value\0with\0nulls");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_count() {
        let count = ConfigRef::count();
        assert!(count > 0, "DuckDB should have configuration options");
    }

    #[test]
    fn test_config_list_available_options() {
        let options = ConfigRef::list_available_options();
        assert!(options.is_ok());

        let options = options.unwrap();
        assert!(!options.is_empty(), "Should have available config options");

        // Check that we got valid option names and descriptions
        for (name, desc) in options.iter().take(5) {
            assert!(!name.is_empty(), "Option name should not be empty");
            assert!(!desc.is_empty(), "Option description should not be empty");
        }

        println!("First few DuckDB config options:");
        for (name, desc) in options.iter().take(3) {
            println!("  {}: {}", name, desc);
        }
    }

    #[test]
    fn test_config_get_flag() {
        // Test getting the first config option
        let first_option = ConfigRef::get_config_flag(0);
        assert!(first_option.is_ok());
        assert!(first_option.unwrap().is_some());

        // Test getting an invalid index
        let invalid_option = ConfigRef::get_config_flag(999999);
        assert!(invalid_option.is_ok());
        assert!(invalid_option.unwrap().is_none());
    }
}
