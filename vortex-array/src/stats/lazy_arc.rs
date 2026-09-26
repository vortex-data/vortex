// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A lazily set, shareable `Arc<T>` that takes one pointer.

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;

/// Null until first use, then an `Arc<T>` shared by clones.
///
/// Takes 8 bytes, like `Arc<T>`, so arrays keep their layout while stats stay lazy.
pub(super) struct LazyArc<T> {
    // Null, or a pointer from `Arc::into_raw` that owns one strong count.
    ptr: AtomicPtr<T>,
}

impl<T> LazyArc<T> {
    /// Wraps an existing `Arc`.
    pub(super) fn from_arc(value: Arc<T>) -> Self {
        Self {
            ptr: AtomicPtr::new(Arc::into_raw(value).cast_mut()),
        }
    }

    pub(super) fn get(&self) -> Option<&T> {
        // SAFETY: a non-null pointer owns a strong count that lives until `drop`.
        unsafe { self.ptr.load(Ordering::Acquire).as_ref() }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ptr.load(Ordering::Acquire).is_null()
    }

    /// Returns the value, installing `init()` if there is none. Concurrent callers race to
    /// install; the losers drop their value.
    pub(super) fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        if let Some(value) = self.get() {
            return value;
        }

        let new = Arc::into_raw(Arc::new(init())).cast_mut();
        match self
            .ptr
            .compare_exchange(ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire)
        {
            // SAFETY: we just installed `new`, which owns a strong count.
            Ok(_) => unsafe { &*new },
            Err(existing) => {
                // SAFETY: `new` was never published, so we still own it; `existing` is live.
                unsafe {
                    drop(Arc::from_raw(new));
                    &*existing
                }
            }
        }
    }

    /// Returns true if both hold the same allocation.
    pub(super) fn ptr_eq(&self, other: &Self) -> bool {
        let this = self.ptr.load(Ordering::Acquire);
        !this.is_null() && this == other.ptr.load(Ordering::Acquire)
    }
}

impl<T> Default for LazyArc<T> {
    fn default() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

impl<T> Clone for LazyArc<T> {
    fn clone(&self) -> Self {
        let ptr = self.ptr.load(Ordering::Acquire);
        if !ptr.is_null() {
            // SAFETY: `ptr` is live because `self` owns a strong count.
            unsafe { Arc::increment_strong_count(ptr) };
        }
        Self {
            ptr: AtomicPtr::new(ptr),
        }
    }
}

impl<T> Drop for LazyArc<T> {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() {
            // SAFETY: we own one strong count for `ptr`.
            unsafe { drop(Arc::from_raw(ptr)) };
        }
    }
}

// SAFETY: `LazyArc<T>` behaves like `Option<Arc<T>>`, which is `Send` when `T: Send + Sync`.
unsafe impl<T: Send + Sync> Send for LazyArc<T> {}
// SAFETY: `LazyArc<T>` behaves like `Option<Arc<T>>`, which is `Sync` when `T: Send + Sync`.
unsafe impl<T: Send + Sync> Sync for LazyArc<T> {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn lazy_until_first_use() {
        let lazy = LazyArc::<u32>::default();
        assert!(lazy.is_empty());
        assert_eq!(*lazy.get_or_init(|| 7), 7);
        assert_eq!(*lazy.get_or_init(|| 8), 7);
        assert_eq!(lazy.get(), Some(&7));
    }

    #[test]
    fn clones_share_the_allocation() {
        let lazy = LazyArc::from_arc(Arc::new(1u32));
        let clone = lazy.clone();
        assert!(lazy.ptr_eq(&clone));
        assert!(!LazyArc::<u32>::default().ptr_eq(&LazyArc::default()));
    }

    #[test]
    fn concurrent_init_installs_one_value() {
        let lazy = Arc::new(LazyArc::<usize>::default());
        let values = (0..8)
            .map(|i| {
                let lazy = Arc::clone(&lazy);
                thread::spawn(move || *lazy.get_or_init(|| i))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap_or_default())
            .collect::<Vec<_>>();

        assert!(values.iter().all(|value| Some(value) == lazy.get()));
    }
}
