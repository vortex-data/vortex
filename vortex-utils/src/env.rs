// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Environment variable utilities for testing.

use dashmap::DashMap;
use parking_lot::Mutex;
use vortex_error::vortex_panic;

/// Global registry of locks per environment variable key.
///
/// Each mutex is lazily created and leaked to get a 'static lifetime.
/// This is acceptable for test utilities that live for the process duration.
static ENV_LOCKS: std::sync::LazyLock<DashMap<&'static str, &'static Mutex<()>>> =
    std::sync::LazyLock::new(DashMap::new);

/// Get or create a static mutex for the given key.
fn get_or_create_lock(key: &'static str) -> &'static Mutex<()> {
    *ENV_LOCKS
        .entry(key)
        .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))))
}

/// RAII guard to set/remove an environment variable for the duration of a scope.
///
/// Removes the variable when dropped, ensuring test isolation.
///
/// This guard holds a mutex lock for the specific environment variable key,
/// ensuring that only one guard can exist for a given key at a time. This
/// prevents tests from accidentally having overlapping guards for the same
/// env var.
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
/// let _guard2 = EnvVarGuard::remove("OTHER_VAR");
/// assert!(std::env::var("OTHER_VAR").is_err());
/// ```
///
/// # Panics
///
/// Panics if a guard already exists for the same environment variable key
/// (detected via mutex lock contention).
///
/// # Safety
///
/// Environment variable modification is inherently unsafe in multi-threaded contexts.
/// This guard is intended for use in tests that are run serially or where env var
/// races are acceptable. The per-key locking ensures that the same env var isn't
/// modified concurrently by multiple guards.
pub struct EnvVarGuard {
    key: &'static str,
    /// We store this to ensure the mutex stays locked for our lifetime.
    /// The () is just a dummy value - we only care about the lock.
    #[expect(dead_code)]
    lock_guard: parking_lot::MutexGuard<'static, ()>,
}

/// Timeout for waiting on an env var lock.
const LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

impl EnvVarGuard {
    /// Acquire the lock for this key, waiting up to 10 seconds.
    ///
    /// If another guard holds the lock, this will wait for it to be released.
    /// If the lock isn't released within 10 seconds, this panics to avoid deadlocks.
    fn acquire_lock(key: &'static str) -> parking_lot::MutexGuard<'static, ()> {
        let mutex = get_or_create_lock(key);
        match mutex.try_lock_for(LOCK_TIMEOUT) {
            Some(guard) => guard,
            None => vortex_panic!(
                "EnvVarGuard: timed out after {LOCK_TIMEOUT:?} waiting for environment variable '{key}'. \
                 This likely indicates a deadlock - ensure guards for the same key are properly scoped \
                 taken in lexicographical order and dropped before acquiring a new one."
            ),
        }
    }

    /// Set an environment variable for the duration of this guard's lifetime.
    pub fn set(key: &'static str, value: &str) -> Self {
        let lock_guard = Self::acquire_lock(key);

        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::set_var(key, value);
        }

        Self { key, lock_guard }
    }

    /// Remove an environment variable for the duration of this guard's lifetime.
    pub fn remove(key: &'static str) -> Self {
        let lock_guard = Self::acquire_lock(key);

        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::remove_var(key);
        }

        Self { key, lock_guard }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::remove_var(self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_remove() {
        let key = "VORTEX_TEST_ENV_VAR_SET";

        // Initially not set
        assert!(std::env::var(key).is_err());

        {
            let _guard = EnvVarGuard::set(key, "test_value");
            assert_eq!(std::env::var(key).unwrap(), "test_value");
        }

        // After guard drops, var is removed
        assert!(std::env::var(key).is_err());
    }

    #[test]
    fn test_remove() {
        let key = "VORTEX_TEST_ENV_VAR_REMOVE";

        // Set it first
        unsafe {
            std::env::set_var(key, "initial");
        }

        {
            let _guard = EnvVarGuard::remove(key);
            assert!(std::env::var(key).is_err());
        }

        // After guard drops, var is still removed
        assert!(std::env::var(key).is_err());
    }

    /// Test that a second guard waits for the first to be released.
    #[test]
    fn test_second_guard_waits_for_first() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::Ordering;
        use std::thread;
        use std::time::Duration;

        let key = "VORTEX_TEST_ENV_VAR_WAIT";
        let second_acquired = Arc::new(AtomicBool::new(false));
        let second_acquired_clone = Arc::clone(&second_acquired);

        // First guard in main thread
        let _guard1 = EnvVarGuard::set(key, "first");
        assert_eq!(std::env::var(key).unwrap(), "first");

        // Spawn thread that will wait for the lock
        let handle = thread::spawn(move || {
            let _guard2 = EnvVarGuard::set(key, "second");
            second_acquired_clone.store(true, Ordering::SeqCst);
            assert_eq!(std::env::var(key).unwrap(), "second");
        });

        // Give the thread time to start waiting
        thread::sleep(Duration::from_millis(50));

        // Second guard should NOT have acquired yet (still waiting)
        assert!(!second_acquired.load(Ordering::SeqCst));

        // Drop the first guard - this should allow the second to proceed
        drop(_guard1);

        // Wait for second thread to complete
        handle.join().unwrap();

        // Now the second guard should have acquired and set the value
        assert!(second_acquired.load(Ordering::SeqCst));
    }
}
