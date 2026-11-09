// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One account of the bytes that scans hold, shared by every run that charges it.
//!
//! A [`MemoryBudget`] counts the bytes of reads that work has asked for and of batches that
//! nobody has consumed yet. Runs that share a budget stop running ahead when it is full, and
//! continue when a consumer releases bytes. The limit is soft: the work that the consumer waits
//! for always runs, so a full budget never stops a scan.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Waker;

use parking_lot::Mutex;

/// The bytes that scans may hold before they stop running ahead.
pub struct MemoryBudget {
    limit: u64,
    used: AtomicU64,
    /// Woken when bytes are released, so a driver that stopped on a full budget runs again.
    waiters: Mutex<Vec<Waker>>,
}

impl MemoryBudget {
    /// A budget of `limit` bytes.
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicU64::new(0),
            waiters: Mutex::new(Vec::new()),
        })
    }

    /// The bytes that the budget allows.
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// The bytes charged and not yet released.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Whether the charged bytes reach the limit.
    pub fn is_full(&self) -> bool {
        self.used() >= self.limit
    }

    /// Charges `bytes`, which are released when the returned charge is dropped.
    pub fn charge(self: &Arc<Self>, bytes: u64) -> MemoryCharge {
        self.used.fetch_add(bytes, Ordering::Relaxed);
        MemoryCharge {
            budget: Arc::clone(self),
            bytes,
        }
    }

    /// Registers `cx` to be woken when bytes are released.
    pub fn wake_on_release(&self, cx: &Context<'_>) {
        let mut waiters = self.waiters.lock();
        if !waiters.iter().any(|waker| waker.will_wake(cx.waker())) {
            waiters.push(cx.waker().clone());
        }
    }

    fn release(&self, bytes: u64) {
        let was_full = self.used.fetch_sub(bytes, Ordering::Relaxed) >= self.limit;
        if !was_full {
            return;
        }

        let waiters = std::mem::take(&mut *self.waiters.lock());
        for waker in waiters {
            waker.wake();
        }
    }
}

/// Bytes charged to a [`MemoryBudget`], released when this is dropped.
pub struct MemoryCharge {
    budget: Arc<MemoryBudget>,
    bytes: u64,
}

impl MemoryCharge {
    /// The bytes this charge holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Charges `bytes` more to the same budget.
    pub fn add(&mut self, bytes: u64) {
        self.budget.used.fetch_add(bytes, Ordering::Relaxed);
        self.bytes += bytes;
    }
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Wake;
    use std::task::Waker;

    use super::MemoryBudget;

    #[derive(Default)]
    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn charges_fill_the_budget_until_they_drop() {
        let budget = MemoryBudget::new(100);
        let mut first = budget.charge(60);
        assert!(!budget.is_full());

        first.add(30);
        let second = budget.charge(20);
        assert_eq!(budget.used(), 110);
        assert!(budget.is_full());

        drop(first);
        assert_eq!(budget.used(), 20);
        assert!(!budget.is_full());

        drop(second);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_release_from_a_full_budget_wakes_the_waiters() {
        let budget = MemoryBudget::new(100);
        let counter = Arc::new(CountingWaker::default());
        let waker = Waker::from(Arc::clone(&counter));
        let cx = Context::from_waker(&waker);

        let small = budget.charge(10);
        budget.wake_on_release(&cx);
        drop(small);
        assert_eq!(counter.0.load(Ordering::Relaxed), 0);

        let large = budget.charge(100);
        // One waker registers once.
        budget.wake_on_release(&cx);
        budget.wake_on_release(&cx);
        drop(large);
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    }
}
