// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned row claims travel with queued and running work.

use std::ops::Range;

use vortex_error::VortexExpect;

use super::FileReads;
use super::ReadEntry;
use super::ReadState;

impl ReadState {
    fn remove_claim(&mut self, start: u64) {
        let count = self
            .claims
            .get_mut(&start)
            .vortex_expect("claim is registered");
        *count -= 1;
        if *count == 0 {
            self.claims.remove(&start);
        }
    }

    /// Return evicted entries so transport cancellation runs outside the state lock.
    fn evict(&mut self) -> Vec<ReadEntry> {
        let Some((&watermark, _)) = self.claims.first_key_value() else {
            self.expiry.clear();
            return self.entries.drain().map(|(_, entry)| entry).collect();
        };
        let mut evicted = Vec::new();
        while self
            .expiry
            .first_key_value()
            .is_some_and(|(&end, _)| end <= watermark)
        {
            let (_, targets) = self.expiry.pop_first().vortex_expect("expiry is present");
            for target in targets {
                if let Some(entry) = self.entries.remove(&target) {
                    evicted.push(entry);
                }
            }
        }
        evicted
    }
}

/// An owned reservation for rows that may still read from a file execution.
///
/// Move the claim with work through queues and planner handoffs. Split it before handing off
/// disjoint children; dropping it marks its remaining rows finished or cancelled. Output batches
/// must not own claims, since retaining a batch does not imply any future file reads.
#[must_use = "dropping a row claim releases its rows"]
pub struct RowClaim {
    file: FileReads,
    rows: Range<u64>,
}

impl RowClaim {
    pub(super) fn new(file: FileReads, rows: Range<u64>) -> Self {
        assert!(rows.start <= rows.end, "claim rows must be ordered");
        if !rows.is_empty() {
            *file.state.lock().claims.entry(rows.start).or_default() += 1;
        }
        Self { file, rows }
    }

    /// The file-local rows this claim still reserves.
    pub fn rows(&self) -> &Range<u64> {
        &self.rows
    }

    /// Transfers the prefix ending at `end` into a child claim, keeping the suffix in `self`.
    /// Both registrations change under one lock, so no observer can see a gap in ownership.
    /// Panics if `end` is outside this claim's range.
    pub fn split_to(&mut self, end: u64) -> Self {
        assert!(
            self.rows.contains(&end) || end == self.rows.end,
            "split must lie within the claim"
        );
        let rows = self.rows.start..end;
        {
            let mut state = self.file.state.lock();
            if !self.rows.is_empty() {
                state.remove_claim(self.rows.start);
            }
            if !rows.is_empty() {
                *state.claims.entry(rows.start).or_default() += 1;
            }
            if end < self.rows.end {
                *state.claims.entry(end).or_default() += 1;
            }
            self.rows.start = end;
        }
        Self {
            file: self.file.clone(),
            rows,
        }
    }

    /// Releases rows proven unnecessary, retaining only the given subrange.
    /// Panics if the range is not ordered and contained in this claim.
    pub fn restrict_to(&mut self, rows: Range<u64>) {
        assert!(
            self.rows.start <= rows.start && rows.start <= rows.end && rows.end <= self.rows.end,
            "remaining rows must be contained in the claim"
        );
        if self.rows == rows {
            return;
        }
        let evicted = {
            let mut state = self.file.state.lock();
            if !self.rows.is_empty() {
                state.remove_claim(self.rows.start);
            }
            if !rows.is_empty() {
                *state.claims.entry(rows.start).or_default() += 1;
            }
            self.rows = rows;
            state.evict()
        };
        drop(evicted);
    }

    /// Creates another owner for the same rows, for consumers that overlap rather than split.
    /// Each owner's remaining rows stay registered until that owner releases them.
    pub fn share(&self) -> Self {
        Self::new(self.file.clone(), self.rows.clone())
    }
}

impl Drop for RowClaim {
    fn drop(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let evicted = {
            let mut state = self.file.state.lock();
            state.remove_claim(self.rows.start);
            state.evict()
        };
        drop(evicted);
    }
}
