// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row limits that are only known after filtering.
//!
//! Without a filter, the number of rows each split contributes is known up front, so the limit can
//! be applied while constructing the split tasks. With a filter, the split tasks evaluate their
//! filters concurrently and then claim rows from a shared [`LimitBudget`] before projecting.
//!
//! Ordered scans must return the first `limit` matching rows, so each split waits for its
//! predecessor to claim before claiming itself. Unordered scans may claim in any order.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::channel::oneshot;

/// The number of rows still to be returned by a filtered scan with a limit.
pub struct LimitBudget {
    remaining: AtomicU64,
}

impl LimitBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            remaining: AtomicU64::new(limit),
        }
    }

    /// Whether every row allowed by the limit has been claimed.
    pub fn is_exhausted(&self) -> bool {
        self.remaining.load(Ordering::Acquire) == 0
    }

    /// Claim up to `rows` rows, returning the number of rows granted.
    fn claim(&self, rows: u64) -> u64 {
        let previous = self
            .remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                Some(remaining - remaining.min(rows))
            })
            .unwrap_or_else(|remaining| remaining);
        previous.min(rows)
    }
}

/// A single split's handle on a shared [`LimitBudget`].
pub struct LimitClaim {
    budget: Arc<LimitBudget>,
    /// Resolves once the preceding split has claimed its rows. `None` when claims are unordered,
    /// or for the first split of an ordered scan.
    prev: Option<oneshot::Receiver<()>>,
    /// Signals the following split of an ordered scan that it may claim.
    next: Option<oneshot::Sender<()>>,
}

impl LimitClaim {
    /// Create the claims for `count` splits sharing `budget`.
    ///
    /// When `ordered`, claims resolve in split order, so every claim must eventually be driven
    /// (or dropped) for the claims after it to make progress.
    pub fn for_splits(
        budget: &Arc<LimitBudget>,
        ordered: bool,
    ) -> impl Iterator<Item = LimitClaim> + use<> {
        let budget = Arc::clone(budget);
        let mut prev = None;
        std::iter::from_fn(move || {
            let (next, prev) = if ordered {
                let (tx, rx) = oneshot::channel();
                (Some(tx), prev.replace(rx))
            } else {
                (None, None)
            };
            Some(LimitClaim {
                budget: Arc::clone(&budget),
                prev,
                next,
            })
        })
    }

    pub fn budget(&self) -> &Arc<LimitBudget> {
        &self.budget
    }

    /// Claim up to `rows` rows once the preceding split, if any, has claimed.
    pub async fn claim(self, rows: u64) -> u64 {
        // A dropped predecessor (cancelled or failed) no longer holds a claim, so proceed.
        if !self.budget.is_exhausted()
            && let Some(prev) = self.prev
        {
            let _canceled = prev.await;
        }
        let granted = self.budget.claim(rows);
        if let Some(next) = self.next {
            let _receiver_dropped = next.send(());
        }
        granted
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;

    use super::LimitBudget;
    use super::LimitClaim;

    #[test]
    fn claims_never_exceed_budget() {
        let budget = Arc::new(LimitBudget::new(5));
        let granted: Vec<u64> = LimitClaim::for_splits(&budget, false)
            .take(3)
            .map(|claim| block_on(claim.claim(3)))
            .collect();

        assert_eq!(granted, [3, 2, 0]);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn ordered_claims_resolve_in_split_order() {
        let budget = Arc::new(LimitBudget::new(4));
        let mut claims: Vec<_> = LimitClaim::for_splits(&budget, true).take(2).collect();
        let second = claims.pop().expect("two claims");
        let first = claims.pop().expect("two claims");

        let second = std::thread::spawn(move || block_on(second.claim(10)));
        assert_eq!(block_on(first.claim(3)), 3);
        assert_eq!(second.join().expect("claim thread panicked"), 1);
    }

    #[test]
    fn dropped_predecessor_does_not_block() {
        let budget = Arc::new(LimitBudget::new(4));
        let mut claims: Vec<_> = LimitClaim::for_splits(&budget, true).take(2).collect();
        let second = claims.pop().expect("two claims");
        drop(claims);

        assert_eq!(block_on(second.claim(10)), 4);
    }
}
