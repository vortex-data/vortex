// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Environment variable utilities for testing.

/// RAII guard to set/remove an environment variable for the duration of a scope.
///
/// Removes the variable when dropped, ensuring test isolation.
///
/// # Example
///
/// ```
/// use vortex_utils::env::EnvVarGuard;
///
/// // Set an env var for the duration of this scope
/// let _guard = EnvVarGuard::set("MY_TEST_VAR", "1");
/// assert_eq!(std::env::var("MY_TEST_VAR").ok(), Some("1".to_string()));
///
/// // Or remove an env var
/// let _guard2 = EnvVarGuard::remove("MY_TEST_VAR");
/// assert!(std::env::var("MY_TEST_VAR").is_err());
/// ```
///
/// # Safety
///
/// Environment variable modification is inherently unsafe in multi-threaded contexts.
/// This guard is intended for use in tests that are run serially or where env var
/// races are acceptable.
pub struct EnvVarGuard(&'static str);

impl EnvVarGuard {
    /// Set an environment variable for the duration of this guard's lifetime.
    pub fn set(key: &'static str, value: &str) -> Self {
        // SAFETY: Tests are run serially or we accept env var race risk.
        unsafe {
            std::env::set_var(key, value);
        }
        Self(key)
    }

    /// Remove an environment variable for the duration of this guard's lifetime.
    pub fn remove(key: &'static str) -> Self {
        // SAFETY: Tests are run serially or we accept env var race risk.
        unsafe {
            std::env::remove_var(key);
        }
        Self(key)
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: See above.
        unsafe {
            std::env::remove_var(self.0);
        }
    }
}
