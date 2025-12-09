// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Environment variable utilities for testing.

use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use arcref::ArcRef;
use dashmap::DashMap;
use parking_lot::ArcMutexGuard;
use parking_lot::Mutex;
use vortex_error::vortex_panic;

/// Global registry of locks per environment variable key.
static ENV_LOCKS: LazyLock<DashMap<ArcRef<str>, Arc<Mutex<()>>>> = LazyLock::new(DashMap::new);

/// Timeout for waiting on an env var lock.
const LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// Get or create a mutex for the given key.
fn get_or_create_lock(key: &ArcRef<str>) -> Arc<Mutex<()>> {
    // Fast path: check if mutex already exists
    if let Some(mutex) = ENV_LOCKS.get(key) {
        return mutex.clone();
    }

    // Slow path: insert new mutex
    ENV_LOCKS
        .entry(key.clone())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
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
/// # Blocking Behavior
///
/// If another guard holds the lock for the same key, this will wait up to 10 seconds
/// for it to be released. If the timeout expires, it panics to avoid deadlocks.
///
/// # Safety
///
/// Environment variable modification is inherently unsafe in multi-threaded contexts.
/// This guard is intended for use in tests. The per-key locking ensures that the
/// same env var isn't modified concurrently by multiple guards.
pub struct EnvVarGuard {
    key: ArcRef<str>,
    /// We store this to ensure the mutex stays locked for our lifetime.
    #[allow(dead_code)]
    lock_guard: ArcMutexGuard<parking_lot::RawMutex, ()>,
}

impl EnvVarGuard {
    /// Acquire the lock for this key, waiting up to 10 seconds.
    ///
    /// If another guard holds the lock, this will wait for it to be released.
    /// If the lock isn't released within 10 seconds, this panics to avoid deadlocks.
    #[allow(clippy::panic)]
    fn acquire_lock(key: &ArcRef<str>) -> ArcMutexGuard<parking_lot::RawMutex, ()> {
        let mutex = get_or_create_lock(key);
        match mutex.try_lock_arc_for(LOCK_TIMEOUT) {
            Some(guard) => guard,
            None => vortex_panic!(
                "EnvVarGuard: timed out after {:?} waiting for environment variable '{}'. \
                 This likely indicates a deadlock - ensure guards for the same key are \
                 properly scoped and dropped before acquiring a new one.",
                LOCK_TIMEOUT,
                key
            ),
        }
    }

    /// Set an environment variable for the duration of this guard's lifetime.
    pub fn set(key: impl Into<ArcRef<str>>, value: &str) -> Self {
        let key: ArcRef<str> = key.into();
        let lock_guard = Self::acquire_lock(&key);

        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::set_var(&*key, value);
        }

        Self { key, lock_guard }
    }

    /// Remove an environment variable for the duration of this guard's lifetime.
    pub fn remove(key: impl Into<ArcRef<str>>) -> Self {
        let key: ArcRef<str> = key.into();
        let lock_guard = Self::acquire_lock(&key);

        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::remove_var(&*key);
        }

        Self { key, lock_guard }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: We hold an exclusive lock for this key.
        unsafe {
            std::env::remove_var(&*self.key);
        }
        // lock_guard is dropped here, releasing the mutex
    }
}
