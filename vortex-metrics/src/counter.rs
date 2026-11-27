// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// A commutative value that can only be increased, and starts at 0 on initialization.
#[derive(Clone, Debug)]
pub struct Counter(Arc<AtomicU64>);

impl Counter {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Adds `value` to the counter
    pub fn add(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Release);
    }

    /// Returns the latest value stored in the counter
    pub fn value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::Counter;

    #[test]
    fn counter_initial_add_and_clone_share_state() {
        let counter = Counter::new();
        assert_eq!(counter.value(), 0);

        counter.add(0);
        assert_eq!(counter.value(), 0);

        counter.add(5);
        assert_eq!(counter.value(), 5);

        let cloned = counter.clone();
        cloned.add(7);
        assert_eq!(counter.value(), 12);
        assert_eq!(cloned.value(), 12);
    }

    #[test]
    fn counter_concurrent_adds() {
        let counter = Counter::new();
        thread::scope(|scope| {
            for _ in 0..8 {
                let c = counter.clone();
                scope.spawn(move || {
                    for _ in 0..1_000 {
                        c.add(3);
                    }
                });
            }
        });
        assert_eq!(counter.value(), 8 * 1_000 * 3);
    }
}
