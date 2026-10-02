// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row limits that are only known after filtering.
//!
//! Without a filter, the number of rows each split contributes is known up front, so the limit can
//! be applied while constructing the split tasks. With a filter, the split tasks evaluate their
//! filters concurrently and then claim rows from a shared [`LimitBudget`] before projecting.
//!
//! Unordered scans may return any `limit` matching rows, so claims are taken from an atomic
//! counter in whatever order the splits finish filtering.
//!
//! Ordered scans must return the first `limit` matching rows, so split `i` is granted
//! `min(matches_i, limit - sum(matches_j for j < i))`. Rather than serializing the splits, each
//! split publishes its match count and decides from bounds on its predecessors' matches: the
//! contiguous prefix of published counts gives a lower bound, and the unfiltered row counts of the
//! remaining predecessors give an upper bound. A split only waits when those bounds straddle the
//! limit, which is at most the few splits in flight around the limit boundary.

use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Poll;
use std::task::Waker;

use parking_lot::Mutex;
use vortex_error::VortexExpect;

/// The rows still to be returned by a filtered scan with a limit.
pub struct LimitBudget {
    limit: u64,
    exhausted: AtomicBool,
    claims: Claims,
}

enum Claims {
    Unordered {
        remaining: AtomicU64,
    },
    Ordered {
        /// `split_rows_prefix[i]` is the number of rows selected, before filtering, by the splits
        /// before split `i`.
        split_rows_prefix: Box<[u64]>,
        state: Mutex<OrderedState>,
    },
}

struct OrderedState {
    /// The published match count of each split.
    matches: Vec<Option<u64>>,
    /// `matches_prefix[i]` is the total match count of the splits before split `i`, for every
    /// split in the contiguous published prefix.
    matches_prefix: Vec<u64>,
    /// Claims waiting for the published prefix to advance.
    waiters: Vec<Waker>,
}

impl OrderedState {
    fn published_prefix_len(&self) -> usize {
        self.matches_prefix.len() - 1
    }

    fn published_prefix_matches(&self) -> u64 {
        *self
            .matches_prefix
            .last()
            .vortex_expect("matches_prefix starts with 0")
    }
}

impl LimitBudget {
    /// Create a budget of `limit` rows shared by splits selecting `split_rows` rows each before
    /// filtering.
    pub fn new(limit: u64, ordered: bool, split_rows: &[u64]) -> Arc<Self> {
        let claims = if ordered {
            let split_rows_prefix = std::iter::once(0)
                .chain(split_rows.iter().scan(0u64, |sum, &rows| {
                    *sum = sum.saturating_add(rows);
                    Some(*sum)
                }))
                .collect();
            Claims::Ordered {
                split_rows_prefix,
                state: Mutex::new(OrderedState {
                    matches: vec![None; split_rows.len()],
                    matches_prefix: vec![0],
                    waiters: Vec::new(),
                }),
            }
        } else {
            Claims::Unordered {
                remaining: AtomicU64::new(limit),
            }
        };
        Arc::new(Self {
            limit,
            exhausted: AtomicBool::new(limit == 0),
            claims,
        })
    }

    /// Whether no split that has not yet been granted rows can be granted any.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }

    /// Record the match count of an ordered split, advancing the published prefix.
    fn publish(&self, split: usize, matches: u64) {
        let Claims::Ordered { state, .. } = &self.claims else {
            return;
        };
        let waiters = {
            let mut state = state.lock();
            state.matches[split] = Some(matches);
            let before = state.published_prefix_len();
            while let Some(&Some(next)) = state.matches.get(state.published_prefix_len()) {
                let sum = state.published_prefix_matches().saturating_add(next);
                state.matches_prefix.push(sum);
            }
            if state.published_prefix_len() == before {
                return;
            }
            if state.published_prefix_matches() >= self.limit {
                self.exhausted.store(true, Ordering::Release);
            }
            std::mem::take(&mut state.waiters)
        };
        waiters.into_iter().for_each(Waker::wake);
    }

    /// Claim rows for split `split`, which matched `matches` rows, returning the rows granted.
    async fn claim(&self, split: usize, matches: u64) -> u64 {
        match &self.claims {
            Claims::Unordered { remaining } => {
                let previous = remaining
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                        Some(remaining - remaining.min(matches))
                    })
                    .unwrap_or_else(|remaining| remaining);
                if previous <= matches {
                    self.exhausted.store(true, Ordering::Release);
                }
                previous.min(matches)
            }
            Claims::Ordered {
                split_rows_prefix,
                state,
            } => {
                self.publish(split, matches);
                poll_fn(|cx| {
                    let mut state = state.lock();
                    let prefix_len = state.published_prefix_len();
                    if prefix_len >= split {
                        let before = state.matches_prefix[split];
                        return Poll::Ready(matches.min(self.limit.saturating_sub(before)));
                    }
                    // Lower and upper bounds on the matches of the splits before this one.
                    let lower = state.published_prefix_matches();
                    if lower >= self.limit {
                        return Poll::Ready(0);
                    }
                    let upper = lower
                        .saturating_add(split_rows_prefix[split] - split_rows_prefix[prefix_len]);
                    if upper.saturating_add(matches) <= self.limit {
                        return Poll::Ready(matches);
                    }
                    state.waiters.push(cx.waker().clone());
                    Poll::Pending
                })
                .await
            }
        }
    }
}

/// A single split's claim on a shared [`LimitBudget`].
///
/// Dropping an ordered claim without claiming (for example because its filter failed) publishes
/// zero matches for its split, so the claims after it do not wait forever.
pub struct LimitClaim {
    budget: Arc<LimitBudget>,
    split: usize,
    claimed: bool,
}

impl LimitClaim {
    /// Create the claim of split `split`, the index of the split among those passed to
    /// [`LimitBudget::new`].
    pub fn new(budget: &Arc<LimitBudget>, split: usize) -> Self {
        Self {
            budget: Arc::clone(budget),
            split,
            claimed: false,
        }
    }

    pub fn budget(&self) -> &Arc<LimitBudget> {
        &self.budget
    }

    /// Claim up to `matches` rows, returning the number of rows granted.
    pub async fn claim(mut self, matches: u64) -> u64 {
        self.claimed = true;
        self.budget.claim(self.split, matches).await
    }
}

impl Drop for LimitClaim {
    fn drop(&mut self) {
        if !self.claimed {
            self.budget.publish(self.split, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::task::Context;
    use std::task::Poll;

    use futures::executor::block_on;
    use futures::task::noop_waker_ref;

    use super::LimitBudget;
    use super::LimitClaim;

    fn poll_claim(fut: std::pin::Pin<&mut impl Future<Output = u64>>) -> Poll<u64> {
        fut.poll(&mut Context::from_waker(noop_waker_ref()))
    }

    #[test]
    fn unordered_claims_never_exceed_budget() {
        let budget = LimitBudget::new(5, false, &[10, 10, 10]);
        let granted: Vec<u64> = (0..3)
            .map(|split| block_on(LimitClaim::new(&budget, split).claim(3)))
            .collect();

        assert_eq!(granted, [3, 2, 0]);
        assert!(budget.is_exhausted());
    }

    #[test]
    fn ordered_claims_grant_first_matches() {
        let budget = LimitBudget::new(4, true, &[10, 10, 10]);
        let mut second = pin!(LimitClaim::new(&budget, 1).claim(10));
        let mut third = pin!(LimitClaim::new(&budget, 2).claim(10));

        // The first split may still match up to 10 rows, so later splits must wait.
        assert_eq!(poll_claim(second.as_mut()), Poll::Pending);
        assert_eq!(poll_claim(third.as_mut()), Poll::Pending);

        assert_eq!(block_on(LimitClaim::new(&budget, 0).claim(3)), 3);
        assert_eq!(poll_claim(second.as_mut()), Poll::Ready(1));
        assert_eq!(poll_claim(third.as_mut()), Poll::Ready(0));
        assert!(budget.is_exhausted());
    }

    #[test]
    fn ordered_claims_within_bounds_do_not_wait() {
        let budget = LimitBudget::new(100, true, &[10, 10, 10]);
        // Every predecessor together selects at most 20 rows, so the limit cannot be reached.
        assert_eq!(block_on(LimitClaim::new(&budget, 2).claim(5)), 5);
        assert!(!budget.is_exhausted());
    }

    #[test]
    fn ordered_claims_after_exhausted_prefix_do_not_wait() {
        let budget = LimitBudget::new(4, true, &[10, 10, 10]);
        assert_eq!(block_on(LimitClaim::new(&budget, 0).claim(6)), 4);
        assert!(budget.is_exhausted());
        // Split 1 has not published, but split 0 alone already fills the limit.
        assert_eq!(block_on(LimitClaim::new(&budget, 2).claim(5)), 0);
    }

    #[test]
    fn dropped_claim_does_not_block_successors() {
        let budget = LimitBudget::new(4, true, &[10, 10]);
        drop(LimitClaim::new(&budget, 0));

        assert_eq!(block_on(LimitClaim::new(&budget, 1).claim(10)), 4);
    }
}
