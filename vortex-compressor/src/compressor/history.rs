// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reuse of scheme decisions across the chunks of a stream.
//!
//! Consecutive chunks of a column usually have similar value distributions, so the scheme that won
//! the search for one chunk usually wins it for the next. A [`ChunkHistory`] records the winner at
//! every compression site. Once the last few searches at a site agree, later chunks skip the search,
//! compress directly with that scheme, and keep the result only if the achieved ratio stays within
//! bounds derived from the searched chunks.

use std::collections::VecDeque;

use parking_lot::Mutex;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scheme::SchemeId;

/// Identifies one compression site within a chunk: the structural path from the chunk root (struct
/// field, list elements, ...) plus the cascade chain at that path.
pub(crate) type SiteKey = (Vec<usize>, Vec<(SchemeId, usize)>);

/// Tuning knobs for [`ChunkHistory`].
#[derive(Debug, Clone, Copy)]
pub struct ChunkHistoryOptions {
    /// Number of consecutive full searches at a site that must pick the same winner before the
    /// winner is reused without searching.
    pub warmup_searches: usize,
    /// How much worse than the worst searched ratio a reused scheme may compress before the site
    /// searches again, as a fraction of that ratio.
    pub ratio_tolerance: f64,
    /// Number of reuses after which the site searches again anyway, to notice when a different
    /// scheme has become better.
    pub recheck_interval: usize,
}

impl Default for ChunkHistoryOptions {
    fn default() -> Self {
        Self {
            warmup_searches: 3,
            ratio_tolerance: 0.2,
            recheck_interval: 16,
        }
    }
}

/// A scheme decision that a site can apply without searching.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plan {
    /// The scheme to apply, or `None` to leave the array canonical.
    pub(crate) scheme: Option<SchemeId>,
    /// The minimum achieved ratio (`before / after`) for the result to be kept.
    pub(crate) min_ratio: f64,
}

/// Search results and the current plan for one site.
#[derive(Debug, Default)]
struct SiteState {
    /// The most recent searches as `(winner, achieved ratio)`, newest last.
    searches: VecDeque<(Option<SchemeId>, f64)>,
    /// The plan derived from `searches`, if they agree.
    plan: Option<Plan>,
    /// Number of times `plan` has been handed out since it was made.
    reuses: usize,
}

/// Scheme decisions shared by the chunks of one stream.
///
/// Pass the same history to [`CascadingCompressor::compress_with_history`] for every chunk of a
/// column. Chunks of different columns should use different histories.
///
/// [`CascadingCompressor::compress_with_history`]: crate::CascadingCompressor::compress_with_history
#[derive(Debug, Default)]
pub struct ChunkHistory {
    /// The tuning knobs.
    options: ChunkHistoryOptions,
    /// Per-site state.
    sites: Mutex<HashMap<SiteKey, SiteState>>,
}

impl ChunkHistory {
    /// Creates an empty history with the given options.
    pub fn new(options: ChunkHistoryOptions) -> Self {
        Self {
            options,
            sites: Mutex::default(),
        }
    }

    /// Returns the plan for `key` if the site should skip its search, counting it as a reuse.
    pub(crate) fn plan(&self, key: &SiteKey) -> Option<Plan> {
        let mut sites = self.sites.lock();
        let site = sites.get_mut(key)?;
        let plan = site.plan?;
        if site.reuses >= self.options.recheck_interval {
            site.plan = None;
            return None;
        }
        site.reuses += 1;
        Some(plan)
    }

    /// Discards everything known about `key` after a reused scheme fell outside its bounds, so the
    /// site warms up from scratch.
    pub(crate) fn invalidate(&self, key: &SiteKey) {
        self.sites.lock().remove(key);
    }

    /// Records the outcome of a full search at `key`.
    pub(crate) fn record_search(&self, key: &SiteKey, winner: Option<SchemeId>, ratio: f64) {
        let warmup = self.options.warmup_searches.max(1);
        let mut sites = self.sites.lock();
        let site = sites.entry(key.clone()).or_default();

        site.searches.push_back((winner, ratio));
        while site.searches.len() > warmup {
            site.searches.pop_front();
        }

        if site.searches.len() == warmup && site.searches.iter().all(|(w, _)| *w == winner) {
            let worst = site
                .searches
                .iter()
                .map(|(_, r)| *r)
                .fold(f64::INFINITY, f64::min);
            site.plan = Some(Plan {
                scheme: winner,
                min_ratio: worst * (1.0 - self.options.ratio_tolerance),
            });
            site.reuses = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: SchemeId = SchemeId { name: "a" };
    const B: SchemeId = SchemeId { name: "b" };

    fn key() -> SiteKey {
        (vec![0], Vec::new())
    }

    #[test]
    fn plans_after_agreeing_warmup() {
        let history = ChunkHistory::default();
        history.record_search(&key(), Some(A), 4.0);
        history.record_search(&key(), Some(A), 5.0);
        assert!(history.plan(&key()).is_none());

        history.record_search(&key(), Some(A), 3.0);
        let plan = history.plan(&key()).expect("warmup agreed");
        assert_eq!(plan.scheme, Some(A));
        assert!((plan.min_ratio - 2.4).abs() < 1e-9);
    }

    #[test]
    fn disagreeing_warmup_keeps_searching() {
        let history = ChunkHistory::default();
        history.record_search(&key(), Some(A), 4.0);
        history.record_search(&key(), Some(B), 4.0);
        history.record_search(&key(), Some(A), 4.0);
        assert!(history.plan(&key()).is_none());

        history.record_search(&key(), Some(A), 4.0);
        assert!(history.plan(&key()).is_none());
        history.record_search(&key(), Some(A), 4.0);
        assert_eq!(history.plan(&key()).map(|p| p.scheme), Some(Some(A)));
    }

    #[test]
    fn rechecks_after_interval() {
        let history = ChunkHistory::new(ChunkHistoryOptions {
            warmup_searches: 1,
            recheck_interval: 2,
            ..Default::default()
        });
        history.record_search(&key(), Some(A), 4.0);
        assert!(history.plan(&key()).is_some());
        assert!(history.plan(&key()).is_some());
        assert!(history.plan(&key()).is_none());

        // One agreeing search re-establishes the plan.
        history.record_search(&key(), Some(A), 4.0);
        assert!(history.plan(&key()).is_some());
    }

    #[test]
    fn invalidate_restarts_warmup() {
        let history = ChunkHistory::default();
        for _ in 0..3 {
            history.record_search(&key(), Some(A), 4.0);
        }
        history.invalidate(&key());
        assert!(history.plan(&key()).is_none());
        history.record_search(&key(), Some(A), 4.0);
        assert!(history.plan(&key()).is_none());
    }
}
