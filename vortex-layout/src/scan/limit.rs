// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_mask::Mask;

/// A cloneable row limit shared by all work that can contribute rows to one scan.
///
/// Rows are reserved from a selection mask before projection work is constructed. This keeps
/// rows that cannot be returned out of projection evaluation entirely. When a limit is shared by
/// concurrent unordered partitions, reservation order is completion order, so callers may return
/// any matching rows. Ordered limited scans instead serialize their external partitions before
/// sharing a `RowLimit`, preserving the first matching rows in scan order.
#[derive(Clone)]
pub(crate) struct RowLimit(Arc<AtomicU64>);

impl RowLimit {
    pub(crate) fn new(limit: u64) -> Self {
        Self(Arc::new(AtomicU64::new(limit)))
    }

    /// Reserve rows selected by `mask` and retain only the earliest granted rows in that mask.
    pub(crate) fn limit(&self, mask: Mask) -> Mask {
        let granted = self.take(mask.true_count());
        mask.limit(granted)
    }

    /// Reserve up to `rows` rows, returning how many the remaining budget granted.
    pub(crate) fn take(&self, rows: usize) -> usize {
        let requested = u64::try_from(rows).unwrap_or(u64::MAX);
        usize::try_from(self.reserve(requested)).unwrap_or(usize::MAX)
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }

    fn reserve(&self, requested: u64) -> u64 {
        let mut remaining = self.0.load(Ordering::Relaxed);
        loop {
            let granted = remaining.min(requested);
            match self.0.compare_exchange_weak(
                remaining,
                remaining - granted,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return granted,
                Err(actual) => remaining = actual,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::thread;

    use vortex_mask::Mask;

    use super::RowLimit;

    #[test]
    fn reserve_grants_up_to_the_remaining_budget() {
        let limit = RowLimit::new(5);
        assert_eq!(limit.reserve(3), 3);
        assert!(!limit.is_exhausted());
        // Only two rows remain, so a larger request saturates at what is left.
        assert_eq!(limit.reserve(10), 2);
        assert!(limit.is_exhausted());
        // Once exhausted, further requests grant nothing.
        assert_eq!(limit.reserve(1), 0);
    }

    #[test]
    fn limit_keeps_the_earliest_granted_rows() {
        let limit = RowLimit::new(2);
        // Rows 0, 2, 3, 5 are selected; only the first two survive the budget of 2.
        let mask = Mask::from_iter([true, false, true, true, false, true]);
        let limited = limit.limit(mask);

        assert_eq!(limited.true_count(), 2);
        assert!(limited.value(0));
        assert!(limited.value(2));
        assert!(!limited.value(3));
        assert!(!limited.value(5));
        assert!(limit.is_exhausted());
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_budget() {
        const THREADS: usize = 8;
        const PER_THREAD: u64 = 10_000;
        const LIMIT: u64 = 25_000;

        let limit = RowLimit::new(LIMIT);
        let granted_total = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(THREADS));

        thread::scope(|scope| {
            for _ in 0..THREADS {
                let limit = limit.clone();
                let granted_total = Arc::clone(&granted_total);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    // Start all threads together to maximize contention on the atomic.
                    barrier.wait();
                    let mut local = 0;
                    for _ in 0..PER_THREAD {
                        local += limit.reserve(1);
                    }
                    granted_total.fetch_add(local, Ordering::Relaxed);
                });
            }
        });

        // Total requested (THREADS * PER_THREAD = 80_000) exceeds the budget, so exactly the
        // budget is granted across all threads — no double-grant, over-grant, or lost reservation.
        assert_eq!(granted_total.load(Ordering::Relaxed), LIMIT);
        assert!(limit.is_exhausted());
    }
}
